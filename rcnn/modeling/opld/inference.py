import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F

from utils.data.structures.quad_bbox import QuadBoxes, cat_quadboxes, quadboxes_nms
from rcnn.core.config import cfg


class OPLDPostProcessor(nn.Module):
    def __init__(self, num_points, roi_feat_size, score_weights, nms_th=0.2, num_classes=16, detections_per_img=150):
        super(OPLDPostProcessor, self).__init__()
        self.num_points = num_points
        self.roi_feat_size = roi_feat_size
        self.whole_map_size = self.roi_feat_size * 4
        self.score_weights = score_weights
        self.nms_th = nms_th
        self.num_classes = num_classes
        self.detections_per_img = detections_per_img

    def forward(self, logits, proposals):
        pred = logits['fused']
        result_box, heat_scores = self.get_boxes(proposals[0], pred)
        results = []
        for proposal in proposals:
            result = self.filter_result(result_box, proposal, heat_scores)
            results.append(result)
        return results

    def get_boxes(self, proposals, pred):
        det_bboxes = proposals.bbox
        device = det_bboxes.get_device()
        det_bboxes = det_bboxes.cpu()

        R, C, H, W = pred.shape
        assert det_bboxes.shape[0] == pred.shape[0]
        assert H == W == self.whole_map_size
        assert C == self.num_points

        pred = pred.sigmoid().cpu()

        pred = pred.view(R * C, H * W)
        pred_scores, pred_position = pred.max(dim=1)
        xs = pred_position % W
        ys = pred_position // W
        # reshape to (num_rois, num_points)
        pred_scores, xs, ys = tuple(map(lambda x: x.view(R, C), [pred_scores, xs, ys]))

        # get expanded pos_bboxes
        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        x1 = (det_bboxes[:, 0, None] - widths / 2)
        y1 = (det_bboxes[:, 1, None] - heights / 2)
        
        # map the point to the absolute coordinates
        abs_xs = (xs.float() + 0.5) / W * widths * 2 + x1
        abs_ys = (ys.float() + 0.5) / H * heights * 2 + y1

        quad_x1 = (abs_xs[:, 0, None])
        quad_y1 = (abs_ys[:, 0, None])
        quad_x2 = (abs_xs[:, 1, None])
        quad_y2 = (abs_ys[:, 1, None])
        quad_x3 = (abs_xs[:, 2, None])
        quad_y3 = (abs_ys[:, 2, None])
        quad_x4 = (abs_xs[:, 3, None])
        quad_y4 = (abs_ys[:, 3, None])

        quad_res = torch.cat([quad_x1, quad_y1, quad_x2, quad_y2, quad_x3, quad_y3, quad_x4, quad_y4], dim=1)

        heat_scores = torch.mean(pred_scores[:, :4], dim=1, keepdim=False).cuda(device=device)

        if cfg.OPLD.USE_CPP:
            assert self.num_points == 5, "CPP requires additional center point prediction!"
            ctr_x = (abs_xs[:, 4])
            ctr_y = (abs_ys[:, 4])
            quad_res = center_post_process(quad_res, widths, heights, ctr_x, ctr_y)
        quad_res = quad_res.cuda(device=device)
        return quad_res, heat_scores

    def filter_result(self, bbox_res, proposal, heat_scores):
        cls_scores = proposal.get_field("scores")
        cls_weight, heat_weight = self.score_weights
        scores = cls_weight * cls_scores + heat_weight * heat_scores
        device = scores.device
        labels = proposal.get_field("labels")
        result = []
        number_of_detections = 0
        for j in range(1, self.num_classes):
            inds = (labels == j).nonzero().view(-1)
            boxes_j = bbox_res[inds, :]
            if boxes_j.size()[0] == 0:
                continue
            scores_j = scores[inds]
            quadboxes_for_class = QuadBoxes(boxes_j, proposal.size)
            quadboxes_for_class.add_field("scores", scores_j)
            quadboxes_for_class = quadboxes_nms(quadboxes_for_class, self.nms_th)
            num_labels = len(quadboxes_for_class)
            number_of_detections += num_labels
            quadboxes_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(quadboxes_for_class)

        result = cat_quadboxes(result)

        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def post_processor():
    num_points = cfg.OPLD.NUM_POINTS
    roi_feat_size = cfg.OPLD.ROI_FEAT_SIZE
    score_weights = cfg.OPLD.SCORE_WEIGHTS
    nms_th = cfg.OPLD.NMS_THRESH
    num_classes = cfg.MODEL.NUM_CLASSES
    detections_per_img = cfg.OPLD.DETECTIONS_PER_IMG

    postprocessor = OPLDPostProcessor(
        num_points,
        roi_feat_size,
        score_weights,
        nms_th,
        num_classes,
        detections_per_img
    )
    return postprocessor


def center_post_process(bboxes, width, height, ctr_x, ctr_y):
    width = width.squeeze(-1)
    height = height.squeeze(-1)
    area = width * height

    ctr_1_x = (bboxes[:, 0] + bboxes[:, 4]) / 2
    ctr_1_y = (bboxes[:, 1] + bboxes[:, 5]) / 2
    ctr_2_x = (bboxes[:, 2] + bboxes[:, 6]) / 2
    ctr_2_y = (bboxes[:, 3] + bboxes[:, 7]) / 2

    judges = area < 15625
    n = torch.as_tensor([10 if judge else 15 for judge in judges], dtype=torch.float)

    c_tl_x = ctr_x - width / n
    c_br_x = ctr_x + width / n
    c_tl_y = ctr_y - height / n
    c_br_y = ctr_y + height / n

    keep_1 = (c_tl_x < ctr_1_x) & (ctr_1_x < c_br_x) & (c_tl_y < ctr_1_y) & (ctr_1_y < c_br_y)
    keep_2 = (c_tl_x < ctr_2_x) & (ctr_2_x < c_br_x) & (c_tl_y < ctr_2_y) & (ctr_2_y < c_br_y)
    keep = keep_1 & keep_2

    keep = np.array(keep)
    keep = np.logical_not(keep)
    index = keep.nonzero()[0]

    bbox_res_new = bboxes.view(-1, 4, 2)
    bbox_res_new = np.array(bbox_res_new)
    for ind in index:
        after_adjust = torch.from_numpy(cv2.boxPoints(cv2.minAreaRect(bbox_res_new[ind]))).view(-1)
        bboxes[ind, :] = after_adjust
    return bboxes
