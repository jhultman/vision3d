from copy import deepcopy
import numpy as np
import torch
from torch import nn

from typing import List

from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

from .bev import BEVFeatureGatherer
from .roi_grid_pool import RoiGridPool
from .backbone import CNN_FACTORY, VoxelFeatureExtractor
from .proposal import ProposalLayer
from .refinement import RefinementLayer


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
        self.vfe = VoxelFeatureExtractor()
        self.cnn = CNN_FACTORY[cfg.CNN](cfg)
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

    def sample_keypoints(self, points):
        """
        fps expects points shape (B, N, 3)
        fps returns indices shape (B, K)
        gather expects features shape (B, C, N)
        """
        points = points[..., :3].contiguous()
        indices = furthest_point_sample(points, self.cfg.NUM_KEYPOINTS)
        keypoints = gather_operation(points.transpose(1, 2).contiguous(), indices)
        keypoints = keypoints.transpose(1, 2).contiguous()
        return keypoints

    def pointnets(self, cnn_out, keypoint_xyz):
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

    def point_feature_extract(self, item, cnn_features, bev_map):
        points_split = torch.split(item['points'], [3, 1], dim=-1)
        cnn_features = [points_split] + cnn_features
        point_features = self.pointnets(cnn_features, item['keypoints'])
        bev_features = self.bev_gatherer(bev_map, item['keypoints'])
        point_features = torch.cat(point_features + [bev_features], dim=1)
        return point_features

    def forward(self, item, proposals_only=False):
        """
        TODO: Document intermediate tensor shapes.
        """
        item['keypoints'] = self.sample_keypoints(item['points'])
        features = self.vfe(item['features'], item['occupancy'])
        cnn_features, bev_map = self.cnn(features, item['coordinates'], item['batch_size'])
        proposal_scores, proposal_boxes = self.proposal_layer(bev_map)
        if proposals_only:
            item.update(dict(proposal_scores=proposal_scores, proposal_boxes=proposal_boxes))
            return item
        point_features = self.point_feature_extract(item, cnn_features, bev_map)
        pooled_features = self.roi_grid_pool(proposals, item['keypoints'], point_features)
        predictions, scores_detections = self.refinement_layer(proposals, pooled_features)
        return predictions
