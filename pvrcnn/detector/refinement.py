import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


class RefinementLayer(nn.Module):
    """Use pooled features to refine proposals."""

    def __init__(self, cfg):
        super(RefinementLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        channels = cfg.REFINEMENT.MLPS + [cfg.NUM_CLASSES * (cfg.BOX_DOF + 1)]
        mlp = MLP(channels, bias=True, bn=False, relu=[True, False])
        return mlp

    def inference(self, points, features):
        boxes, scores = self(points, features)
        scores = F.softmax(scores, dim=-1)
        positive = 1 - scores[..., -1:]
        _, indices = torch.topk(positive, k=self.cfg.PROPOSAL.TOPK, dim=1)
        indices = indices.expand(-1, -1, self.cfg.NUM_CLASSES)
        box_indices = indices[..., None].expand(-1, -1, -1, self.cfg.BOX_DOF)
        scores = scores.gather(1, indices)
        boxes = boxes.gather(1, box_indices)
        return boxes, scores, indices

    def forward(self, points, features, proposal_boxes):
        features = features.permute(0, 2, 1)
        refinements = self.mlp(features)
        box_deltas, scores = self.reorganize_predictions(refinements)
        boxes = self.apply_refinements(box_deltas, proposal_boxes)
        return boxes, scores
