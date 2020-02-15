import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


class ProposalLoss(nn.Module):

    def __init__(self, cfg):
        super(ProposalLoss, self).__init__()
        self.cfg = cfg
        self.anchors = self.cfg.ANCHORS

    def compute_label(self, predictions, boxes, class_idx, mask):
        """
        predictions of shape (B, N, D)
        groundtruth of shape (B, M, D)
        """
        center_pred, center_true = predictions[..., 0:3], boxes[..., 0:3]
        wlh_pred, wlh_true = predictions[..., 3:6], boxes[..., 3:6]
        yaw_pred, yaw_true = predictions[..., 6:7], boxes[..., 6:7]

    def forward(self, predictions, boxes, class_idx, mask):
        raise NotImplementedError


class ProposalLayer(nn.Module):
    """
    Use keypoint features to generate 3D box proposals.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        channels = cfg.PROPOSAL.MLPS + [cfg.NUM_CLASSES * (cfg.BOX_DOF + 1)]
        mlp = MLP(channels, bias=True, bn=False, relu=[True, False])
        return mlp

    def inference(self, points, features):
        boxes, scores = self(points, features)
        _, indices = torch.topk(scores, k=self.cfg.PROPOSAL.TOPK, dim=1)
        scores = scores.gather(1, indices)
        boxes = boxes.gather(1, indices.expand(-1, -1, boxes.shape[-1]))
        return boxes, scores

    def reorganize_proposals(self, proposals):
        B, N, _ = proposals.shape
        proposals = proposals.view(B, N, self.cfg.NUM_CLASSES, -1)
        boxes, scores = proposals.split([self.cfg.BOX_DOF, 1], dim=-1)
        return boxes, scores.squeeze(-1)

    def forward(self, points, features):
        features = features.permute(0, 2, 1)
        proposals = self.mlp(features)
        boxes, scores = self.reorganize_proposals(proposals)
        scores = F.softmax(scores, dim=-1)
        return boxes, scores
