import torch
from torch import nn


class RefinementTargetAssigner(nn.Module):
    """
    TODO: Remove batch support to simplify implementation.
    TODO: Complete rewrite.
    """

    def __init__(self, cfg):
        super(RefinementTargetAssigner, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.NUM_CLASSES
        anchor_sizes = [anchor['wlh'] for anchor in cfg.ANCHORS]
        anchor_radii = [anchor['radius'] for anchor in cfg.ANCHORS]
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

    def fill_negatives(self, targets_cls):
        """TODO: REFINEMENT_NUM_NEGATIVES needs rethinking."""
        M = self.cfg.TRAIN.REFINEMENT_NUM_NEGATIVES
        (B, N, _) = targets_cls.shape
        inds = torch.randint(N, (B, M), dtype=torch.long)
        targets_cls[:, inds, -2] = 1
        targets_cls[:, inds, -1] = 0

    def fill_positives(self, targets_cls, inds):
        i, j, k = inds
        targets_cls[i, j, k] = 1
        targets_cls[i, j, -2:] = 0

    def fill_ambiguous(self, targets_cls):
        """Disables positives matched to multiple classes."""
        ambiguous = targets_cls.int().sum(2) > 1
        targets_cls[ambiguous][:, :-1] = 0
        targets_cls[ambiguous][:, -1] = 1

    def make_cls_targets(self, inds, shape, device):
        """
        Note that some negatives will be overwritten by positives.
        Last two indices are background and ignore, respectively.
        Uses one-hot encoding.
        """
        B, N, _ = shape
        targets_cls = torch.zeros(
            (B, N, self.num_classes + 2), dtype=torch.long, device=device)
        targets_cls[..., -1] = 1
        self.fill_negatives(targets_cls)
        self.fill_positives(targets_cls, inds)
        self.fill_ambiguous(targets_cls)
        return targets_cls

    def make_reg_targets(self, inds, boxes, keypoints, anchor_sizes):
        i, j, k = inds
        B, N, _ = keypoints.shape
        targets_reg = torch.zeros(
            (B, N, self.num_classes, 7), dtype=torch.float32, device=keypoints.device)
        box_centers, box_sizes, box_angles = boxes.split([3, 3, 1], dim=-1)
        targets_reg[i, j, k, 0:3] = box_centers[k] - keypoints[i, j]
        targets_reg[i, j, k, 3:6] = (box_sizes[k] - anchor_sizes[k]) / anchor_sizes[k]
        targets_reg[i, j, k, 6:7] = box_angles[k]
        return targets_reg

    def match_keypoints(self, boxes, keypoints, anchor_radii, class_ids, box_counts):
        """Find keypoints within spherical radius of ground truth center."""
        box_centers, box_sizes, box_angles = boxes.split([3, 3, 1], dim=-1)
        distances = torch.norm(keypoints[:, :, None, :] - box_centers, dim=-1)
        in_radius = distances < anchor_radii[class_ids]
        in_radius &= self.batch_correspondence_mask(box_counts, keypoints.device)
        return in_radius.nonzero().t()

    def get_targets(self, item):
        box_counts = [b.shape[0] for b in item['boxes']]
        boxes = torch.cat(item['boxes'], dim=0)
        class_ids = torch.cat(item['class_ids'], dim=0)
        keypoints = item['keypoints']
        device = keypoints.device
        anchor_sizes = self.anchor_sizes.to(device)
        anchor_radii = self.anchor_radii.to(device)
        i, j, k = self.match_keypoints(boxes, keypoints, anchor_radii, class_ids, box_counts)
        inds = (i, j, class_ids[k])
        targets_cls = self.make_cls_targets(inds, keypoints.shape, device)
        targets_reg = self.make_reg_targets(inds, boxes, keypoints, anchor_sizes)
        return targets_cls, targets_reg

    def forward(self, item):
        raise NotImplementedError
