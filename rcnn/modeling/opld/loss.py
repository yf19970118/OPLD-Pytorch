import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
from utils.data.structures.boxlist_ops import boxlist_iou
from rcnn.utils.matcher import Matcher
from rcnn.utils.misc import cat, keep_only_positive_boxes
from rcnn.core.config import cfg


def make_mapping_heatmap(positive_proposals, gt_proposals, num_points, roi_feat_size, pos_radius):
    size = int(np.sqrt(num_points))
    whole_map_size = roi_feat_size * 4
    pos_bboxes = positive_proposals.bbox
    pos_gt_bboxes = gt_proposals.quad_bbox
    device = pos_gt_bboxes.device
    assert pos_bboxes.shape[0] == pos_gt_bboxes.shape[0]

    # expand pos_bboxes to 2x of original size          xyxy
    x1 = pos_bboxes[:, 0] - (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
    y1 = pos_bboxes[:, 1] - (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
    x2 = pos_bboxes[:, 2] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
    y2 = pos_bboxes[:, 3] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
    pos_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    pos_bbox_ws = (pos_bboxes[:, 2] - pos_bboxes[:, 0]).unsqueeze(-1)
    pos_bbox_hs = (pos_bboxes[:, 3] - pos_bboxes[:, 1]).unsqueeze(-1)

    num_rois = pos_bboxes.shape[0]
    map_size = whole_map_size
    targets = torch.zeros((num_rois, num_points, map_size, map_size), dtype=torch.float)

    radius = pos_radius
    radius2 = radius ** 2
    for i in range(num_rois):
        # ignore small bboxes
        if pos_bbox_ws[i] <= size or pos_bbox_hs[i] <= size:
            continue
        for j in range(num_points):
            # get the coordinates from QuadBoxes
            point_x = pos_gt_bboxes[i, j * 2]
            point_y = pos_gt_bboxes[i, j * 2 + 1]

            # map the point in absolute coordinates to the whole heat map
            cx = int((point_x - pos_bboxes[i, 0]) / pos_bbox_ws[i] * map_size)
            cy = int((point_y - pos_bboxes[i, 1]) / pos_bbox_hs[i] * map_size)

            for x in range(cx - radius, cx + radius + 1):
                for y in range(cy - radius, cy + radius + 1):
                    if 0 <= x < map_size and 0 <= y < map_size:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= radius2:
                            targets[i, j, y, x] = 1
    targets = torch.as_tensor(targets).to(device=device)
    return targets


class HGridLossComputation(object):
    def __init__(self, loss_weight, proposal_matcher, pos_radius, num_points, roi_feat_size):
        self.loss_weight = loss_weight
        self.proposal_matcher = proposal_matcher
        self.pos_radius = pos_radius
        self.num_points = num_points
        self.roi_feat_size = roi_feat_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_quad_with_fields(["labels"])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        all_positive_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # targets are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            positive_proposals = proposals_per_image[positive_inds]
            gt_proposals = matched_targets[positive_inds]

            targets = make_mapping_heatmap(
                positive_proposals, gt_proposals, self.num_points, self.roi_feat_size, self.pos_radius
            )
            positive_proposals.add_field("targets", targets)
            all_positive_proposals.append(positive_proposals)

        return all_positive_proposals

    def resample(self, proposals, targets):
        # get all positive proposals (for single image on per GPU)
        positive_proposals = keep_only_positive_boxes(proposals)
        # resample for getting targets or matching new IoU
        positive_proposals = self.prepare_targets(positive_proposals, targets)

        self.positive_proposals = positive_proposals

        all_num_positive_proposals = 0
        for positive_proposals_per_image in positive_proposals:
            all_num_positive_proposals += len(positive_proposals_per_image)
        if all_num_positive_proposals == 0:
            positive_proposals = [proposals[0][:1]]
        return positive_proposals

    def __call__(self, logits):
        targets = [proposals_per_img.get_field("targets") for proposals_per_img in self.positive_proposals]
        targets = cat(targets, dim=0).float()

        if targets.numel() == 0:
            return logits['fused'].sum() * 0

        loss_fused = self.loss_weight * F.binary_cross_entropy_with_logits(logits['fused'], targets)
        loss_unfused = self.loss_weight * F.binary_cross_entropy_with_logits(logits['unfused'], targets)
        loss = loss_fused + loss_unfused
        return loss


def loss_evaluator():
    matcher = Matcher(
        cfg.FAST_RCNN.FG_IOU_THRESHOLD,
        cfg.FAST_RCNN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    loss_weight = cfg.OPLD.LOSS_WEIGHT
    pos_radius = cfg.OPLD.POS_RADIUS
    num_points = cfg.OPLD.NUM_POINTS
    roi_feat_size = cfg.OPLD.ROI_FEAT_SIZE

    evaluator = HGridLossComputation(
        loss_weight,
        matcher,
        pos_radius,
        num_points,
        roi_feat_size,
    )
    return evaluator
