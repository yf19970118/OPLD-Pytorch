from models.ops import _C


def box_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.
    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    return _C.box_iou_rotated(boxes1, boxes2)


def box_iou_polygon(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in
    (x1, y1, x2, y2, x3, y3, x4, y4) format.
    Arguments:
        boxes1 (Tensor[N, 8])
        boxes2 (Tensor[M, 8])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    return _C.box_iou_polygon(boxes1, boxes2)
