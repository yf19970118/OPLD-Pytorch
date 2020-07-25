import polyiou
import time
import torch
import numpy as np


def calcoverlaps(BBGT_keep, bb):
    overlaps = []
    for index, GT in enumerate(BBGT_keep):
        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
        overlaps.append(overlap)
    return overlaps


def caloverlap(BBGT, BBDET):
    matrix = np.zeros((len(BBGT), len(BBDET)), dtype=np.float)
    for index, det in enumerate(BBDET):
        matrix[:, index] = calcoverlaps(BBGT, anchors[index])
    return matrix


if __name__ == '__main__':
    # TODO: improve the precision, the results seems like a little diffrerent from polyiou.cpp
    # , may caused by use float not double.
    anchors = np.array([
        [1, 1, 2, 10, 0, 6, 7, 8],
        #                           [1, 30, 3, 1, np.pi/16],
        #                           [1000, 1000, 60, 60, 0],

                        ],
                       dtype=np.float)
    anchors = np.repeat(anchors, 1000, axis=0)
    gt_boxes = np.array([
        [2, 1, 2, 10, 0, 6, 7, 8],
        #                           [1, 30, 3, 1, np.pi/16 + np.pi/2],
        #                           [1010, 1010, 3, 3, 0],

                         ], dtype=np.float)
    gt_boxes = np.repeat(gt_boxes, 1000, axis=0)
    # anchors = np.array([[1, 1, 200, 100, 0]],
    #                    dtype=np.float32)
    # gt_boxes = np.array([[2, 1, 200, 100, 0],
    #                      ], dtype=np.float32)
    # anchors = np.array([[1, 30, 3, 1, np.pi/16]],
    #                    dtype=np.float32)
    # gt_boxes = np.array([[1, 30, 3, 1, np.pi/16 + np.pi/2],
    #                      ], dtype=np.float32)
    star_time = time.time()
    g = torch.as_tensor(gt_boxes, dtype=torch.float).reshape(-1, 8)
    b = torch.as_tensor(anchors, dtype=torch.float).reshape(-1, 8)
    # overlap = polyiou.iou_poly(polyiou.VectorDouble(gt_boxes[8]), polyiou.VectorDouble(anchors[8]))
    overlaps = caloverlap(gt_boxes, anchors)
    end_time = time.time()
    print(end_time - star_time)
    print(overlaps)
