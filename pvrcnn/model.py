import numpy as np
import torch
from torch import nn

import spconv
from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pointnet2_utils import furthest_point_sample

from pvrcnn.config import PvrcnnConfig
from pvrcnn.cnn import CNN_3D


class PV_RCNN(nn.Module):
    """
    For each feature volume stride, convert keypoint locations to
    continuous voxel index coordinates. Then fetch voxels within ball query.
    Raw input points are treated as an additional stride-1 voxel stage.
    """

    def __init__(self, cfg):
        super(PV_RCNN, self).__init__()
        self.pnets = self.build_pointnets(cfg)
        self.voxel_generator, grid_shape = self.build_voxel_generator(cfg)
        self.cnn = CNN_3D(grid_shape=grid_shape, cfg=cfg)
        self.cfg = cfg

    def build_voxel_generator(self, cfg):
        """Voxel-grid is reversed XYZ -> ZYX and padded in Z-axis."""
        voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=cfg.voxel_size,
            point_cloud_range=cfg.grid_bounds,
            max_voxels=cfg.max_voxels,
            max_num_points=cfg.max_num_points,
        )
        grid_shape = np.r_[voxel_generator.grid_size[::-1]] + [1, 0, 0]
        return voxel_generator, grid_shape

    def build_pointnets(self, cfg):
        """Copy channel list because PointNet modifies it in-place."""
        pnets = []
        for i in range(len(cfg.mlps)):
            pnets += [PointnetSAModuleMSG(
                npoint=-1, radii=cfg.radii[i], nsamples=cfg.nsamples[i],
                mlps=cfg.mlps[i].copy(), use_xyz=True,
            )]
        return nn.Sequential(*pnets)

    def voxelize(self, points):
        """
        Compute sparse voxel grid.
        :points_in np.ndarray of shape (Np, 4)
        :points_out FloatTensor of shape (Np, 4)
        :features FloatTensor of shape (Nv, 1)
        :coordinates IntTensor of shape (Nv, 4)
        :voxel_population IntTensor of shape (Nv, 1)
        """
        features, coordinates, voxel_population = self.voxel_generator.generate(points)
        coordinates = np.pad(coordinates, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        from_numpy = lambda x: torch.from_numpy(x).cuda()
        points, features, coordinates, voxel_population = map(
            from_numpy, (points, features, coordinates, voxel_population))
        features = features.view(-1, self.cfg.max_num_points * self.cfg.C_in)
        return points, features, coordinates, voxel_population

    def sample_keypoints(self, xyz, point_features):
        """
        Sample keypoints from raw pointcloud. Assumes unit batch size.
        :xyz FloatTensor of shape (N, 3).
        :point_features FloatTensor of shape (N, C).
        :return tuple of \
            FloatTensor of shape (n_keypoints, 3),
            FloatTensor of shape (n_keypoints, C)
        """
        xyz = xyz.unsqueeze(0).contiguous()
        indices = furthest_point_sample(xyz, self.cfg.n_keypoints).squeeze(0).long()
        keypoint_xyz = xyz[:, indices].squeeze(0)
        keypoint_features = point_features[indices]
        return keypoint_xyz, keypoint_features

    def pnet_forward(self, cnn_out, keypoint_xyz):
        """
        Call PointNet modules to gather keypoint features
        from the intermediate 3D CNN feature maps.
        """
        pnet_out = []
        for i, pnet in enumerate(self.pnets):
            voxel_coords, voxel_features = cnn_out[i]
            voxel_coords = voxel_coords.unsqueeze(0).contiguous()
            voxel_features = voxel_features.t().unsqueeze(0).contiguous()
            out = pnet(voxel_coords, voxel_features, keypoint_xyz)[1]
            pnet_out += [out]
        return pnet_out

    def bev_forward(self, features, coordinates, keypoint_xyz):
        """
        Project 3D voxel grid to XY-plane and gather keypoint features.
        """
        raise NotImplementedError

    def forward(self, points):
        """
        TODO: Document intermediate tensor shapes.
        TODO: Add BEV point aggregation.
        """
        points, features, coordinates, voxel_population = self.voxelize(points)
        cnn_out = self.cnn(features, coordinates, batch_size=1)
        point_xyz, point_features = torch.split(points, [3, 1], dim=-1)
        cnn_out = [(point_xyz, point_features)] + cnn_out
        keypoint_xyz, keypoint_features = self.sample_keypoints(point_xyz, point_features)
        keypoint_xyz = keypoint_xyz.unsqueeze(0).contiguous()
        pnet_out = self.pnet_forward(cnn_out, keypoint_xyz)
        return pnet_out


def main():
    cfg = PvrcnnConfig()
    net = PV_RCNN(cfg).cuda()
    points = np.random.uniform(0, 50, size=(120000, cfg.C_in)).astype(np.float32)
    out = net(points)


if __name__ == '__main__':
    main()
