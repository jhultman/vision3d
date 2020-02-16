import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


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
        scores = F.softmax(scores, dim=-1)
        _, indices = torch.topk(scores, k=self.cfg.PROPOSAL.TOPK, dim=1)
        box_indices = indices[..., None].expand(-1, -1, -1, self.cfg.BOX_DOF)
        scores = scores.gather(1, indices)
        boxes = boxes.gather(1, box_indices)
        return boxes, scores, indices

    def reorganize_proposals(self, proposals):
        B, N, _ = proposals.shape
        proposals = proposals.view(B, N, self.cfg.NUM_CLASSES, -1)
        boxes, scores = proposals.split([self.cfg.BOX_DOF, 1], dim=-1)
        return boxes, scores.squeeze(-1)

    def forward(self, points, features):
        features = features.permute(0, 2, 1)
        proposals = self.mlp(features)
        boxes, scores = self.reorganize_proposals(proposals)
        return boxes, scores
