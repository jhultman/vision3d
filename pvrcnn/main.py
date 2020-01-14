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


class CNN_3D(nn.Module):
    """
    Simple placeholder sparse 3D CNN with four blocks:

        block_0: [1600, 1280, 41] -> [1600, 1280, 41]
        block_1: [1600, 1280, 41] -> [800, 640, 21]
        block_2: [800, 640, 21]   -> [400, 320, 11]
        block_3: [400, 320, 11]   -> [200, 160, 6]

    Returns feature volumes strided 1x, 2x, 4x, 8x.
    """

    def __init__(self, C_in, grid_shape, cfg):
        super(CNN_3D, self).__init__()
        self.blocks = spconv.SparseSequential(
            spconv.SparseConv3d(C_in, 16, 3, 1, padding=0, bias=False),
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

        TODO: Ensure ijk indices consistent with xyz metric coordinates.
        """
        feature, index = volume.features, volume.indices
        voxel_size = self.base_voxel_size * (2 ** stride)
        xyz = index[..., 1:].float() * voxel_size
        xyz = (xyz + self.voxel_offset)
        return feature, xyz

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
        self.pnet = PointnetSAModuleMSG(
            npoint=-1, radii=[0.1, 0.5], nsamples=[16, 32],
            mlps=[[16, 32, 64], [16, 32, 128]], use_xyz=True,
        )
        self.voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=cfg.voxel_size,
            point_cloud_range=cfg.grid_bounds,
            max_voxels=cfg.max_voxels,
            max_num_points=cfg.max_num_points,
        )
        grid_shape = np.r_[self.voxel_generator.grid_size[::-1]] + [1, 0, 0]
        self.cnn = CNN_3D(C_in=cfg.C_in, grid_shape=grid_shape, cfg=cfg)
        self.cfg = cfg

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
        features = features.view(-1, self.cfg.C_in)
        return points, features, coordinates, voxel_population

    def forward(self, points):
        """
        TODO: Document intermediate tensor shapes.
        TODO: Use all feature volume strides of 3D CNN output.
        """
        points, features, coordinates, voxel_population = self.voxelize(points)
        out = self.cnn(features, coordinates, batch_size=1)

        xyz, point_features = torch.split(points, [3, 1], dim=-1)
        out = [(point_features, xyz)] + out

        xyz = xyz.unsqueeze(0).contiguous()
        indices = furthest_point_sample(xyz, cfg.n_keypoints).squeeze(0).long()
        keypoints = points[indices]
        keypoints_xyz, keypoints_features = torch.split(keypoints, [3, 1], dim=-1)
        voxel_features_i, voxel_coords_i = out[2]

        voxel_coords_i = voxel_coords_i.unsqueeze(0).contiguous()
        voxel_features_i = voxel_features_i.unsqueeze(0).permute(0, 2, 1).contiguous()
        keypoints_xyz = keypoints_xyz.unsqueeze(0).contiguous()

        _, out = self.pnet(voxel_coords_i, voxel_features_i, keypoints_xyz)
        return out


def main():
    cfg = PvrcnnConfig()
    net = PV_RCNN(cfg).cuda()
    points = np.fromfile(cfg.sample_fpath, dtype=np.float32).reshape(-1, cfg.C_in)
    out = net(points)


if __name__ == '__main__':
    main()
