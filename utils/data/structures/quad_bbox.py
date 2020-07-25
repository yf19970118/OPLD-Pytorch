import numpy as np
import torch
import cv2
import math
from utils.data.structures.bounding_box import BoxList
from models.ops import nms_polygon as _box_nms_polygon
from models.ops import box_iou_polygon as _box_iou_polygon

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class QuadBoxes(object):
    def __init__(self, bbox, image_size, mode="xyxy"):      # 只有在回归时使用xywh
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 8 and bbox.size(-1) != 10:
            raise ValueError(
                "last dimension of bbox should have a size of 8 or 10, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.device = device
        self.quad_bbox = bbox
        self.mode = mode
        self.size = image_size
        self.bbox = self.quad_bbox_to_hor_bbox(mode)
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        bbox = QuadBoxes(self.quad_bbox, self.size, mode)
        bbox._copy_extra_fields(self)
        return bbox

    def quad_bbox_to_hor_bbox(self, mode):
        bbox = torch.zeros((self.quad_bbox.shape[0], 4))
        if self.quad_bbox.shape[0] == 0:
            return bbox.to(self.device)
        x1, _ = torch.min(self.quad_bbox[:, 0:8:2], 1)
        y1, _ = torch.min(self.quad_bbox[:, 1:8:2], 1)
        x2, _ = torch.max(self.quad_bbox[:, 0:8:2], 1)
        y2, _ = torch.max(self.quad_bbox[:, 1:8:2], 1)
        if mode == 'xyxy':
            bbox[:, 0] = x1
            bbox[:, 1] = y1
            bbox[:, 2] = x2
            bbox[:, 3] = y2
        elif mode == 'xywh':
            TO_REMOVE = 1
            width = x2 - x1 + TO_REMOVE
            height = y2 - y1 + TO_REMOVE
            bbox[:, 0] = x1 + width // 2
            bbox[:, 1] = y1 + height // 2
            bbox[:, 2] = width
            bbox[:, 3] = height
        else:
            raise RuntimeError("Should not be here")
        return bbox.to(self.device)

    def keep_quad(self, remove_empty=False):
        quad_bbox = self.quad_bbox
        if quad_bbox.size(-1) == 10:
            x1, y1, x2, y2, x3, y3, x4, y4, _, _ = quad_bbox.chunk(10, dim=1)
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = quad_bbox.chunk(8, dim=1)

        delta = determinant(x3 - x1, x2 - x4, y3 - y1, y2 - y4)
        intersection = (delta >= 1e-6) | (delta <= -1e-6)

        namenda = determinant(x2 - x1, x2 - x4, y2 - y1, y2 - y4) / delta
        intersection = intersection & ((namenda <= 1) & (namenda >= 0))

        miu = determinant(x3 - x1, x2 - x1, y3 - y1, y2 - y1) / delta
        intersection = intersection & ((miu <= 1) & (miu >= 0))
        keep = torch.as_tensor(intersection).squeeze(-1)
        quad = self[keep]
        if remove_empty:
            return quad.remove_small_boxes(0)
        return quad

    def remove_small_boxes(self, min_size):
        bbox = self.bbox
        widths = bbox[:, 2]
        heights = bbox[:, 3]
        keep = (widths > min_size) & (heights > min_size)
        return self[keep]

    def resize(self, size):
        if size == self.size:
            return self
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        assert ratios[0] == ratios[1]
        ratio = ratios[0]
        quad_bbox = self.quad_bbox
        scaled_box = quad_bbox * ratio

        quad = QuadBoxes(scaled_box, size, self.mode)
        for k, v in self.extra_fields.items():
            if not isinstance(v, (torch.Tensor, np.ndarray, list)):
                v = v.resize(size, *args, **kwargs)
            quad.add_field(k, v)
        return quad

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )
        image_width, image_height = self.size
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = self.quad_bbox.split(1, dim=-1)
        if method == FLIP_LEFT_RIGHT:
            transposed_x1 = image_width - x2
            transposed_x2 = image_width - x1
            transposed_x3 = image_width - x4
            transposed_x4 = image_width - x3
            transposed_x5 = image_width - x5
            transposed_y1 = y2
            transposed_y2 = y1
            transposed_y3 = y4
            transposed_y4 = y3
            transposed_y5 = y5
            transposed_boxes = torch.cat(
                (transposed_x1, transposed_y1, transposed_x2, transposed_y2,
                 transposed_x3, transposed_y3, transposed_x4, transposed_y4, transposed_x5, transposed_y5), dim=-1
            )
        elif method == FLIP_TOP_BOTTOM:
            transposed_x1 = x2
            transposed_x2 = x1
            transposed_x3 = x4
            transposed_x4 = x3
            transposed_x5 = x5
            transposed_y1 = image_height - y2
            transposed_y2 = image_height - y1
            transposed_y3 = image_height - y4
            transposed_y4 = image_height - y3
            transposed_y5 = image_height - y5
            transposed_boxes = torch.cat(
                (transposed_x1, transposed_y1, transposed_x2, transposed_y2,
                 transposed_x3, transposed_y3, transposed_x4, transposed_y4,transposed_x5, transposed_y5), dim=-1
            )
        quad = QuadBoxes(transposed_boxes, self.size, self.mode)
        quad._copy_extra_fields(self)
        return quad

    def rotate(self, theta):
        if theta == 0:
            return self
        quad_bbox = self.quad_bbox
        device = quad_bbox.device
        h, w = self.size
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)

        new_points_list = []
        obj_num = quad_bbox.size()[0]
        for st in range(0, quad_bbox.size()[1], 2):
            points = quad_bbox[:, st:st + 2]
            expand_points = np.concatenate((points, np.ones(shape=(obj_num, 1))), axis=1)
            new_points = np.dot(M, expand_points.T)
            new_points = new_points.T
            new_points_list.append(new_points)
        rotated_quad = np.concatenate(new_points_list, axis=1).astype(int)
        rotated_quad = torch.from_numpy(rotated_quad).to(device)
        quad = QuadBoxes(rotated_quad, self.size, self.mode)
        quad._copy_extra_fields(self)
        return quad
    
    def change_order(self, change_categories):
        quad_bbox = self.quad_bbox
        x1 = self.bbox[:, 0]
        y1 = self.bbox[:, 1]
        classes = self.get_field("labels")
        quad_x = quad_bbox[:, 0:8:2] - x1.view(-1, 1).repeat(1, 4)
        quad_y = quad_bbox[:, 1:8:2] - y1.view(-1, 1).repeat(1, 4)
        distance = torch.sqrt(torch.pow(quad_x, 2) + torch.pow(quad_y, 2))
        small, min_index = torch.min(distance, -1)
        min_index = min_index.view(-1, 1)
        order_index = min_index
        for i in range(1, 4):
            next_index = (min_index + i) % 4
            order_index = torch.cat([order_index, next_index], dim=-1)
        inds = np.in1d(classes, change_categories, invert=False)
        inds = torch.as_tensor(inds)
        index_new = inds.nonzero()
        for i, ins in enumerate(index_new):
            quad_bbox[ins, 0:8:2] = quad_bbox[ins, (order_index[i] * 2)]
            quad_bbox[ins, 1:8:2] = quad_bbox[ins, (order_index[i] * 2 + 1)]
        quad = QuadBoxes(quad_bbox, self.size, self.mode)
        quad._copy_extra_fields(self)
        return quad

    # Tensor-like methods
    def to(self, device):
        quad = QuadBoxes(self.quad_bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            quad.add_field(k, v)
        return quad

    def __getitem__(self, item):
        quad = QuadBoxes(self.quad_bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            quad.add_field(k, v[item])
        return quad

    def __len__(self):
        return self.quad_bbox.shape[0]

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return area

    def copy_with_fields(self, fields, skip_missing=False):     # 直接返回BoxList
        boxlist = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                boxlist.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return boxlist

    def copy_quad_with_fields(self, fields, skip_missing=False):     # 直接返回BoxList
        quad = QuadBoxes(self.quad_bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                quad.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return quad

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_quadboxes(bboxes):
    """
    Concatenates a list of QuadBoxes (having the same image size) into a
    single QuadBoxes
    Arguments:
        bboxes (list[QuadBoxes])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, QuadBoxes) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = QuadBoxes(_cat([bbox.quad_bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def get_rotate_quad(quad, ctr_x, ctr_y, theta):
    """
    when theta > 0， quad will rotate ccw about the center point(ctr_x, ctr_y)
    :param quad: (x1, y1, ..., x4, y4) (n, 8)，Absolute coordinates
           rbbox: (ctr_x, ctr_y, w, h, theta)
    :return: boxes: (rotate_x1, rotate_y1, ..., rotate_x4, rotate_y4) (n, 8)，Absolute coordinates
    """
    device = quad.device
    boxes = torch.zeros_like(quad, dtype=torch.float, device=device)
    ctr_x = ctr_x.view(-1, 1)
    ctr_y = ctr_y.view(-1, 1)
    theta = theta * math.pi / 180
    cos = torch.cos(theta).view(-1, 1)
    sin = torch.sin(theta).view(-1, 1)
    xx = quad[:, 0::2]
    xx -= ctr_x
    yy = quad[:, 1::2]
    yy -= ctr_y
    x = yy * sin + xx * cos + ctr_x
    y = yy * cos - xx * sin + ctr_y
    boxes[:, 0::2] = x
    boxes[:, 1::2] = y
    return boxes


def quadboxes_nms(quadboxes, nms_thresh=0.2):
    dets = quadboxes.quad_bbox
    scores = quadboxes.get_field("scores")
    keep = _box_nms_polygon(dets, scores, nms_thresh)
    quadboxes = quadboxes[keep]
    return quadboxes


def quadboxes_iou(quadboxes1, quadboxes2):
    """
    Given two lists of rotated boxes of size N and M,
    compute the IoU (intersection over union)between __all__ N x M pairs of boxes.
    The box order must be (x_center, y_center, width, height, angle).
    Args:
        boxes1, boxes2 (QuadBoxes):
    Returns:
        Tensor: IoU, sized [N,M].
    """
    if isinstance(quadboxes1, QuadBoxes) and isinstance(quadboxes2, QuadBoxes):
        iou = _box_iou_polygon(quadboxes1.quad_bbox, quadboxes2.quad_bbox)
    else:
        iou = _box_iou_polygon(quadboxes1, quadboxes2)
    return iou
