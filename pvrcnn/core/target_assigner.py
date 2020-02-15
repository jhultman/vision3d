import torch
from torch import nn


class TargetAssigner(nn.Module):

    def __init__(self, cfg):
        super(TargetAssigner, self).__init__()
        self.cfg = cfg
        self.num_classes = len(self.cfg.ANCHORS) + 1
        anchor_sizes = [anchor['wlh'] for anchor in self.cfg.ANCHORS]
        anchor_radii = [anchor['radius'] for anchor in self.cfg.ANCHORS]
        self.anchor_sizes = torch.tensor(anchor_sizes).float()
        self.anchor_radii = torch.tensor(anchor_radii).float()

    def batch_correspondence_mask(self, box_counts, device):
        """
        Trick to ensure boxes not matched to wrong batch index.
        """
        num_boxes, batch_size = sum(box_counts), len(box_counts)
        box_inds = torch.arange(num_boxes)
        box_batch_inds = torch.repeat_interleave(
            torch.arange(batch_size), torch.LongTensor(box_counts))
        mask = torch.full((batch_size, 1, num_boxes),
            False, dtype=torch.bool, device=device)
        mask[box_batch_inds, :, box_inds] = True
        return mask

    def assign_proposal(self, input_dict):
        """
        Simple target assignment algorithm based on Sparse-to-Dense
        Keypoints considered positive if within category-specific
        max spherical radius of box center.

        Regression and class targets are one-hot encoded (per-anchor predictions)

        TODO: Refactor so can reuse for refinement target assignment.
        TODO: Ensure no keypoint assigned to multiple boxes.
        NOTE: num_classes includes background class
        NOTE: background class is encoded as last index of class dimension.
        """
        box_counts = [b.shape[0] for b in input_dict['boxes']]
        boxes = torch.cat(input_dict['boxes'], dim=0)
        class_ids = torch.cat(input_dict['class_ids'], dim=0)
        keypoints = input_dict['keypoints']
        device = keypoints.device
        anchor_sizes = self.anchor_sizes.to(device)
        anchor_radii = self.anchor_radii.to(device)

        box_centers, box_sizes, box_angles = torch.split(boxes, [3, 3, 1], dim=-1)
        distances = torch.norm(keypoints[:, :, None, :] - box_centers, dim=-1)
        in_radius = distances < anchor_radii[class_ids]
        in_radius &= self.batch_correspondence_mask(box_counts, device)

        i, j, k = in_radius.nonzero().t()
        k = class_ids[k]

        B, N, _ = keypoints.shape
        targets_cls = torch.zeros((B, N, self.num_classes), dtype=torch.bool, device=device)
        targets_reg = torch.zeros((B, N, self.num_classes, 7), dtype=torch.float32, device=device)
        targets_cls[i, j, k] = 1
        targets_cls[..., -1] = ~(targets_cls[..., :-1]).any(-1)
        targets_reg[i, j, k, 0:3] = keypoints[i, j] - box_centers[k]
        targets_reg[i, j, k, 3:6] = box_sizes[k] / anchor_sizes[k]
        targets_reg[i, j, k, 6:7] = box_angles[k]
        targets = dict(proposal_cls=targets_cls, proposal_reg=targets_reg)
        return targets

    def forward(self, input_dict):
        """
        TODO: Assign refinement targets.
        """
        input_dict.update(self.assign_proposal(input_dict))
        return input_dict
