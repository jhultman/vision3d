from copy import deepcopy
import numpy as np
import torch
from torch import nn

from typing import List

import spconv
from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

from pvrcnn.config import cfg
from pvrcnn.bev import BEVFeatureGatherer
from pvrcnn.roi_grid_pool import RoiGridPool
from pvrcnn.backbone import SparseCNN, VoxelFeatureExtractor
from pvrcnn.proposal import ProposalLayer
from pvrcnn.refinement import RefinementLayer


class PV_RCNN(nn.Module):
    """
    For each feature volume stride, convert keypoint locations to
    continuous voxel index coordinates. Then fetch voxels within ball query.
    Raw input points are treated as an additional stride-1 voxel stage.
    """

    def __init__(self, cfg):
        super(PV_RCNN, self).__init__()
        self.pnets = self.build_pointnets(cfg)
        self.roi_grid_pool = RoiGridPool(cfg)
        self.voxel_generator, grid_shape = self.build_voxel_generator(cfg)
        self.vfe = VoxelFeatureExtractor()
        self.cnn = SparseCNN(grid_shape, cfg)
        self.bev_gatherer = self.build_bev_gatherer(cfg)
        self.proposal_layer = ProposalLayer(cfg)
        self.refinement_layer = RefinementLayer(cfg)
        self.cfg = cfg

    def build_voxel_generator(self, cfg):
        """Voxel-grid is reversed XYZ -> ZYX and padded in Z-axis."""
        voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=cfg.VOXEL_SIZE,
            point_cloud_range=cfg.GRID_BOUNDS,
            max_voxels=cfg.MAX_VOXELS,
            max_num_points=cfg.MAX_OCCUPANCY,
        )
        grid_shape = np.r_[voxel_generator.grid_size[::-1]] + [1, 0, 0]
        return voxel_generator, grid_shape

    def build_pointnets(self, cfg):
        """Copy list because PointNet modifies it in-place."""
        pnets = []
        for i, mlps in enumerate(cfg.PSA.MLPS):
            pnets += [PointnetSAModuleMSG(
                npoint=-1, radii=cfg.PSA.RADII[i],
                nsamples=cfg.SAMPLES_PN,
                mlps=deepcopy(mlps), use_xyz=True,
            )]
        return nn.Sequential(*pnets)

    def build_bev_gatherer(self, cfg):
        bev = BEVFeatureGatherer(
            cfg, self.cnn.voxel_offset, self.cnn.base_voxel_size)
        return bev

    def generate_batch_voxels(self, points):
        """Voxelize points and tag with batch index."""
        features, coordinates, occupancy = [], [], []
        for i, p in enumerate(points):
            f, c, o = self.voxel_generator.generate(p)
            c = np.pad(c, ((0, 0), (1, 0)), constant_values=i)
            features += [f]; coordinates += [c]; occupancy += [o]
        return map(np.concatenate, (features, coordinates, occupancy))

    def from_numpy(self, x):
        """Make cuda tensor."""
        return torch.from_numpy(x).cuda()

    def voxelize(self, points):
        """
        Compute sparse voxel grid.
        :points_in list of np.ndarrays of shape (Np, 4)
        :points_out FloatTensor of shape (Np, 4)
        :features FloatTensor of shape (Nv, 1)
        :coordinates IntTensor of shape (Nv, 4)
        """
        features, coordinates, occupancy = self.generate_batch_voxels(points)
        points = self.pad_for_batch(points)
        points, features, coordinates, occupancy = \
            map(self.from_numpy, (points, features, coordinates, occupancy))
        features = self.vfe(features, occupancy)
        return points, features, coordinates

    def sample_keypoints(self, points):
        """
        Sample keypoints from raw pointcloud.
            - fps expects points shape (B, N, 3)
            - fps returns indices shape (B, K)
            - gather expects features shape (B, C, N)
        """
        points = points[..., :3].transpose(1, 2).contiguous()
        indices = furthest_point_sample(points, self.cfg.NUM_KEYPOINTS)
        keypoints = gather_operation(points, indices).transpose(1, 2).contiguous()
        return keypoints

    def pnet_forward(self, cnn_out, keypoint_xyz):
        """
        Call PointNets to gather keypoint features from CNN feature volumes.
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz: (B, npoint, 3) tensor of the new features' xyz
        :return (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        pnet_out = []
        for (voxel_xyz, voxel_features), pnet in zip(cnn_out, self.pnets):
            voxel_xyz = voxel_xyz.contiguous()
            voxel_features = voxel_features.transpose(1, 2).contiguous()
            out = pnet(voxel_xyz, voxel_features, keypoint_xyz)[1]
            pnet_out += [out]
        return pnet_out

    def pad_for_batch(self, points: List) -> torch.Tensor:
        """Pad with subsampled points to form dense minibatch."""
        num_points = np.r_[[p.shape[0] for p in points]]
        pad = num_points.max() - num_points
        points_batch = []
        for points_i, pad_i in zip(points, pad):
            idx = np.random.choice(points_i.shape[0], pad_i)
            points_batch += [np.concatenate((points_i, points_i[idx]))]
        points = np.stack(points_batch, axis=0)
        return points

    def forward(self, points):
        """
        TODO: Document intermediate tensor shapes.
        TODO: Use dicts or struct to group elements.
        """
        batch_size = len(points)
        point_lengths = [len(p) for p in points]
        points, features, coordinates = self.voxelize(points)
        cnn_out, final_volume = self.cnn(features, coordinates, batch_size=batch_size)
        cnn_out = [torch.split(points, [3, 1], dim=-1)] + cnn_out
        keypoints_xyz = self.sample_keypoints(points)
        pnet_out = self.pnet_forward(cnn_out, keypoints_xyz)
        bev_out = self.bev_gatherer(final_volume, keypoints_xyz)
        features = torch.cat(pnet_out + [bev_out], dim=1)
        proposals, scores_proposal = self.proposal_layer(keypoints_xyz, features)
        pooled_features = self.roi_grid_pool(proposals, keypoints_xyz, features)
        predictions, scores_detections = self.refinement_layer(proposals, pooled_features)
        return predictions


def make_points(n, cfg):
    points = np.random.uniform(
        0, 50, size=(n, cfg.C_IN)).astype(np.float32)
    return points


def main():
    net = PV_RCNN(cfg).cuda()
    points = [make_points(95000, cfg), make_points(90000, cfg)]
    out = net(points)


if __name__ == '__main__':
    main()
