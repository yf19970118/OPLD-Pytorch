import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from utils.data.structures.bounding_box import BoxList
from rcnn.utils.box_coder import BoxCoder
from rcnn.core.config import cfg


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, num_classes, score_thresh=0.05, box_coder=None, cls_agnostic_bbox_reg=False):
        """
        Arguments:
            num_classes (int)
            score_thresh (float)
            box_coder (BoxCoder)
            cls_agnostic_bbox_reg (bool)
        """
        super(PostProcessor, self).__init__()
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        if cfg.FAST_RCNN.ROI_CLS_LOSS == 'cross_entropy':
            class_prob = F.softmax(class_logits, -1)
        elif cfg.FAST_RCNN.ROI_CLS_LOSS == 'eql':
            class_prob = F.sigmoid(class_logits)
            n = class_prob.size(0)
            dummy_probs = class_prob.new_zeros(n, 1)
            class_prob = torch.cat([dummy_probs, class_prob], dim=1)
        else:
            raise NotImplementedError

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if box_regression is not None:
            if self.cls_agnostic_bbox_reg:
                box_regression = box_regression[:, -4:]
            proposals = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)
        else:
            proposals = concat_boxes.repeat(1, class_prob.shape[1])

        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_bg(boxlist)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_bg(self, boxlist):
        scores = boxlist.get_field("scores")
        device = scores.device
        num_repeat = int(boxlist.bbox.shape[0] / self.num_classes)
        labels = np.tile(np.arange(self.num_classes), num_repeat)
        boxlist.add_field("labels", torch.from_numpy(labels).to(dtype=torch.int64, device=device))
        fg_labels = torch.from_numpy((np.arange(boxlist.bbox.shape[0]) % self.num_classes != 0).astype(int)
                                     ).to(dtype=torch.uint8, device=device)
        inds_all = (scores > self.score_thresh) & (fg_labels > 0)
        return boxlist[inds_all]


def box_post_processor():
    num_classes = cfg.MODEL.NUM_CLASSES
    score_thresh = cfg.FAST_RCNN.SCORE_THRESH
    box_coder = BoxCoder(weights=cfg.FAST_RCNN.BBOX_REG_WEIGHTS)
    cls_agnostic_bbox_reg = cfg.FAST_RCNN.CLS_AGNOSTIC_BBOX_REG

    postprocessor = PostProcessor(
        num_classes,
        score_thresh=score_thresh,
        box_coder=box_coder,
        cls_agnostic_bbox_reg=cls_agnostic_bbox_reg
    )
    return postprocessor
