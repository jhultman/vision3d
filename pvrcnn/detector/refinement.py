import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


class RefinementLayer(nn.Module):
    """
    Uses pooled features to refine proposals.

    TODO: Pass class predictions from proposals since this
        module only predicts confidence.
    TODO: Implement RefinementLoss.
    TODO: Decide if decode box predictions / apply box
        deltas here or elsewhere.
    """

    def __init__(self, cfg):
        super(RefinementLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        """
        TODO: Check if should use bias.
        """
        channels = cfg.REFINEMENT.MLPS + [cfg.BOX_DOF + 1]
        mlp = MLP(channels, bias=True, bn=False, relu=[True, False])
        return mlp

    def inference(self, points, features, proposal_boxes):
        box_deltas, scores = self(points, features, proposal_boxes)
        boxes = self.apply_refinements(box_deltas, proposal_boxes)
        scores = scores.sigmoid()
        positive = 1 - scores[..., -1:]
        _, indices = torch.topk(positive, k=self.cfg.PROPOSAL.TOPK, dim=1)
        indices = indices.expand(-1, -1, self.cfg.NUM_CLASSES)
        box_indices = indices[..., None].expand(-1, -1, -1, self.cfg.BOX_DOF)
        scores = scores.gather(1, indices)
        boxes = boxes.gather(1, box_indices)
        return boxes, scores, indices

    def forward(self, points, features, proposal_boxes):
        refinements = self.mlp(features.permute(0, 2, 1))
        box_deltas, scores = refinements.split(1)
        return box_deltas, scores
