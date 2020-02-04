import numpy as np
import torch
from torch import nn

import spconv
from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pointnet2_utils import furthest_point_sample


class PvrcnnConfig:

    C_in = 4
    n_keypoints = 2048
    strides = [1, 2, 4, 8]
    max_num_points = 5
    max_voxels = 40000
    voxel_size = [0.05, 0.05, 0.1]
    grid_bounds = [0, -40, -3, 64, 40, 1]
    sample_fpath = './sample.bin'

    # PointNet params
    radii = [
        [0.4, 0.8], [0.4, 0.8], [0.8, 1.2], [1.2, 2.4], [2.4, 4.8]
    ]
    nsamples = [[16, 32]] * len(radii)
    mlps = [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]]
    ]


class CNN_3D(nn.Module):
    """
    Simple placeholder sparse 3D CNN with four blocks:

        block_0: [1600, 1280, 41] -> [1600, 1280, 41]
        block_1: [1600, 1280, 41] -> [800, 640, 21]
        block_2: [800, 640, 21]   -> [400, 320, 11]
        block_3: [400, 320, 11]   -> [200, 160, 6]

    Input points within voxels are concatenated along channels.
    Returns feature volumes strided 1x, 2x, 4x, 8x.
    """

    def __init__(self, grid_shape, cfg):
        super(CNN_3D, self).__init__()
        C = cfg.C_in * cfg.max_num_points
        self.blocks = spconv.SparseSequential(
            spconv.SparseConv3d(C, 16, 3, 1, padding=0, bias=False),
            spconv.SparseConv3d(16, 16, 3, 2, padding=1, bias=False),
            spconv.SparseConv3d(16, 32, 3, 2, padding=1, bias=False),
            spconv.SparseConv3d(32, 64, 3, 2, padding=1, bias=False),
        )
        self.grid_shape = grid_shape
        self.base_voxel_size = torch.cuda.FloatTensor(cfg.voxel_size)
        self.voxel_offset = torch.cuda.FloatTensor(cfg.grid_bounds[:3])

    def to_global(self, stride, volume):
        """
        Convert integer voxel indices to metric coordinates.
        voxel_size: length-3 tensor describing size of atomic voxel, accounting for stride.
        voxel_offset: length-3 tensor describing coordinate offset of voxel grid.

        TODO: Ensure ijk indices order consistent with xyz metric coordinates.
        """
        feature, index = volume.features, volume.indices
        voxel_size = self.base_voxel_size * (2 ** stride)
        xyz = index[..., 1:].float() * voxel_size
        xyz = (xyz + self.voxel_offset)
        return xyz, feature

    def forward(self, features, coordinates, batch_size):
        x0 = spconv.SparseConvTensor(
            features, coordinates.int(), self.grid_shape, batch_size,
        )
        x1 = self.blocks[0](x0)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        x4 = self.blocks[3](x3)
        x = [self.to_global(i, x) for i, x in enumerate([x1, x2, x3, x4])]
        return x


class PV_RCNN(nn.Module):
    """
    For each feature volume stride, convert keypoint locations to
    continuous voxel index coordinates. Then fetch voxels within ball query.
    """

    def __init__(self, cfg):
        """
        TODO: Read Pointnet params from config object.
        """
        super(PV_RCNN, self).__init__()
        self.pnets = self.build_pointnets(cfg)
        self.voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=cfg.voxel_size,
            point_cloud_range=cfg.grid_bounds,
            max_voxels=cfg.max_voxels,
            max_num_points=cfg.max_num_points,
        )
        grid_shape = np.r_[self.voxel_generator.grid_size[::-1]] + [1, 0, 0]
        self.cnn = CNN_3D(grid_shape=grid_shape, cfg=cfg)
        self.cfg = cfg

    def build_pointnets(self, cfg):
        pnets = []
        for i in range(len(cfg.strides)):
            pnets += [PointnetSAModuleMSG(
                npoint=-1, radii=cfg.radii[i], nsamples=cfg.nsamples[i],
                mlps=cfg.mlps[i], use_xyz=True,
            )]
        return nn.Sequential(*pnets)

    def voxelize(self, points):
        """
        Compute sparse voxel grid.
        TODO: Ensure feature vectorization is correct.
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
        pnet_out = []
        for i in range(len(self.cfg.strides)):
            voxel_coords, voxel_features = cnn_out[i]
            voxel_coords = voxel_coords.unsqueeze(0).contiguous()
            voxel_features = voxel_features.t().unsqueeze(0).contiguous()
            _, out = self.pnets[i](voxel_coords, voxel_features, keypoint_xyz)
            pnet_out += [out]
        return pnet_out

    def forward(self, points):
        """
        TODO: Document intermediate tensor shapes.
        TODO: Concatenate features from different strides.
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
    points = np.fromfile(cfg.sample_fpath, dtype=np.float32).reshape(-1, cfg.C_in)
    out = net(points)


if __name__ == '__main__':
    main()
