from copy import deepcopy
import numpy as np
import torch
from torch import nn

from typing import List

from pointnet2.pointnet2_modules import PointnetSAModuleMSG

from .bev import BEVFeatureGatherer
from .roi_grid_pool import RoiGridPool
from .backbone import SparseCNN, VoxelFeatureExtractor
from .proposal import ProposalLayer
from .refinement import RefinementLayer


class PV_RCNN(nn.Module):
    """
    For each feature volume stride, convert keypoint locations to
    continuous voxel index coordinates. Then fetch voxels within ball query.
    Raw input points are treated as an additional stride-1 voxel stage.
    """

    def __init__(self, cfg, preprocessor):
        super(PV_RCNN, self).__init__()
        self.preprocessor = preprocessor
        self.pnets = self.build_pointnets(cfg)
        self.roi_grid_pool = RoiGridPool(cfg)
        self.vfe = VoxelFeatureExtractor()
        self.cnn = SparseCNN(preprocessor.grid_shape, cfg)
        self.bev_gatherer = self.build_bev_gatherer(cfg)
        self.proposal_layer = ProposalLayer(cfg)
        self.refinement_layer = RefinementLayer(cfg)
        self.cfg = cfg

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

    def forward(self, input_dict):
        """
        TODO: Document intermediate tensor shapes.
        TODO: Use dicts or struct to group elements.
        """
        input_dict = self.preprocessor(input_dict)
        features = self.vfe(input_dict['features'], input_dict['occupancy'])
        coordinates, batch_size = input_dict['coordinates'], input_dict['batch_size']
        keypoints_xyz, points = input_dict['keypoints'], input_dict['points']
        cnn_out, final_volume = self.cnn(features, coordinates, batch_size=batch_size)
        cnn_out = [torch.split(points, [3, 1], dim=-1)] + cnn_out
        pnet_out = self.pnet_forward(cnn_out, keypoints_xyz)
        bev_out = self.bev_gatherer(final_volume, keypoints_xyz)
        features = torch.cat(pnet_out + [bev_out], dim=1)
        proposals, scores_proposal = self.proposal_layer(keypoints_xyz, features)
        pooled_features = self.roi_grid_pool(proposals, keypoints_xyz, features)
        predictions, scores_detections = self.refinement_layer(proposals, pooled_features)
        return predictions
