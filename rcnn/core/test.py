import cv2
import numpy as np
import pycocotools.mask as mask_util

import torch
from torch.nn import functional as F

from utils.data.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import cat_boxlist, boxlist_nms, \
    boxlist_ml_nms, boxlist_soft_nms, boxlist_box_voting
from rcnn.core.config import cfg


def im_detect_bbox(model, ims):
    box_results = [[] for _ in range(len(ims))]
    features = []
    results, net_imgs_size, conv_features = im_detect_bbox_net(
        model, ims, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
    )
    if cfg.RPN.RPN_ONLY:
        return results, None
    add_results(box_results, results)
    features.append((net_imgs_size, conv_features))

    if cfg.TEST.BBOX_AUG.ENABLED:
        if cfg.TEST.BBOX_AUG.H_FLIP:
            results_hf, net_imgs_size_hf, conv_features_hf = im_detect_bbox_net(
                model, ims, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, True, net_imgs_size
            )
            add_results(box_results, results_hf)
            features.append((net_imgs_size_hf, conv_features_hf))

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
            results_scl, net_imgs_size_scl, conv_features_scl = im_detect_bbox_net(
                model, ims, scale, max_size, False, net_imgs_size
            )
            add_results(box_results, results_scl)
            features.append((net_imgs_size_scl, conv_features_scl))

            if cfg.TEST.BBOX_AUG.H_FLIP:
                results_scl_hf, net_imgs_size_scl_hf, conv_features_scl_hf = im_detect_bbox_net(
                    model, ims, scale, max_size, True, net_imgs_size
                )
                add_results(box_results, results_scl_hf)
                features.append((net_imgs_size_scl_hf, conv_features_scl_hf))

    if cfg.MODEL.HAS_BOX:
        nms_thresh, detections_per_img = get_detection_params()
        box_results = [cat_boxlist(result) for result in box_results]
        box_results = [
            filter_results(
                result, nms_thresh=nms_thresh, detections_per_img=detections_per_img
            ) for result in box_results
        ]
    else:
        box_results = [result[0] for result in box_results]

    return box_results, features


def im_detect_bbox_net(model, ims, target_scale, target_max_size, flip=False, size=None):
    net_imgs_size = []
    results = []
    ims_blob = get_blob(ims, target_scale, target_max_size, flip)
    conv_features, box_results = model.box_net(ims_blob)

    for i, im_result in enumerate(box_results):
        net_img_size = im_result.size
        net_imgs_size.append(net_img_size)
        if flip:
            im_result = im_result.transpose(0)
            if len(cfg.TRAIN.LEFT_RIGHT) > 0:
                scores = im_result.get_field("scores").reshape(-1, cfg.MODEL.NUM_CLASSES)
                boxes = im_result.bbox.reshape(-1, cfg.MODEL.NUM_CLASSES, 4)
                idx = torch.arange(cfg.MODEL.NUM_CLASSES)
                for j in cfg.TRAIN.LEFT_RIGHT:
                    idx[j[0]] = j[1]
                    idx[j[1]] = j[0]
                boxes = boxes[:, idx].reshape(-1, 4)
                scores = scores[:, idx].reshape(-1)
                im_result.bbox = boxes
                im_result.add_field("scores", scores)
        if size:
            im_result = im_result.resize(size[i])
        results.append(im_result)
    return results, net_imgs_size, conv_features


def im_detect_opld(model, rois, features):
    grid_probs = [[] for _ in range(len(rois))]
    aug_idx = 0

    conv_features = features[aug_idx][1]
    results = model.opld_net(conv_features, rois, targets=None)
    #
    # if cfg.TEST.BBOX_AUG.ENABLED and cfg.TEST.GRID_AUG.ENABLED:
    #     if len(rois[0]) == 0:
    #         return results
    #     probs = [result.get_field("grid") for result in results]
    #     add_results(grid_probs, probs)
    #     aug_idx += 1
    #
    #     if cfg.TEST.BBOX_AUG.H_FLIP:
    #         rois_hf = [roi.transpose(0) for roi in rois]
    #         features_hf = features[aug_idx][1]
    #         results_hf = model.grid_net(features_hf, rois_hf, targets=None)
    #         probs_hf = [result_hf.get_field("grid") for result_hf in results_hf]
    #         probs_hf = [torch.flip(prob_hf, dims=(3,)) for prob_hf in probs_hf]
    #         probs_hf = [prob_hf[:, [6, 7, 8, 3, 4, 5, 0, 1, 2], :, :] for prob_hf in probs_hf]
    #         add_results(grid_probs, probs_hf)
    #         aug_idx += 1
    #
    #     for scale in cfg.TEST.BBOX_AUG.SCALES:
    #         rois_scl = [roi.resize(size) for roi, size in zip(rois, features[aug_idx][0])]
    #         features_scl = features[aug_idx][1]
    #         results_scl = model.grid_net(features_scl, rois_scl, targets=None)
    #         probs_scl = [result_scl.get_field("grid") for result_scl in results_scl]
    #         add_results(grid_probs, probs_scl)
    #         aug_idx += 1
    #
    #         if cfg.TEST.BBOX_AUG.H_FLIP:
    #             rois_scl_hf = [roi.resize(size) for roi, size in zip(rois, features[aug_idx][0])]
    #             rois_scl_hf = [roi.transpose(0) for roi in rois_scl_hf]
    #             features_scl_hf = features[aug_idx][1]
    #             results_scl_hf = model.grid_net(features_scl_hf, rois_scl_hf, targets=None)
    #             probs_scl_hf = [result_scl_hf.get_field("grid") for result_scl_hf in results_scl_hf]
    #             probs_scl_hf = [torch.flip(prob_scl_hf, dims=(3,)) for prob_scl_hf in probs_scl_hf]
    #             probs_scl_hf = [prob_scl_hf[:, [6, 7, 8, 3, 4, 5, 0, 1, 2], :, :] for prob_scl_hf in probs_scl_hf]
    #             add_results(grid_probs, probs_scl_hf)
    #             aug_idx += 1
    #
    #     for probs_ts, result in zip(grid_probs, results):
    #         probs_ts = torch.stack(probs_ts, dim=0)
    #         probs_avg = torch.mean(probs_ts, dim=0)
    #         result.add_field("grid", probs_avg)

    return results


def filter_results(boxlist, nms_thresh=0.5, detections_per_img=100):
    num_classes = cfg.MODEL.NUM_CLASSES
    if not cfg.TEST.SOFT_NMS.ENABLED and not cfg.TEST.BBOX_VOTE.ENABLED:
        result = boxlist_ml_nms(boxlist, nms_thresh)
    else:
        boxes = boxlist.bbox
        scores = boxlist.get_field("scores")
        labels = boxlist.get_field("labels")
        result = []
        for j in range(1, num_classes):  # skip the background
            inds = (labels == j).nonzero().view(-1)
            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class_old = boxlist_for_class
            if cfg.TEST.SOFT_NMS.ENABLED:
                boxlist_for_class = boxlist_soft_nms(
                    boxlist_for_class,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=nms_thresh,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                boxlist_for_class = boxlist_nms(boxlist_for_class, nms_thresh)
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED and boxes_j.shape[0] > 0:
                boxlist_for_class = boxlist_box_voting(
                    boxlist_for_class,
                    boxlist_for_class_old,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field("labels", torch.full((num_labels,), j, dtype=torch.int64, device=scores.device))
            result.append(boxlist_for_class)
        result = cat_boxlist(result)

    # Limit to max_per_image detections **over all classes**
    number_of_detections = len(result)
    if number_of_detections > detections_per_img > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - detections_per_img + 1)
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


def add_results(all_results, results):
    for i in range(len(all_results)):
        all_results[i].append(results[i])


def get_blob(ims, target_scale, target_max_size, flip):
    ims_processed = []
    for im in ims:
        if flip:
            im = im[:, ::-1, :]
        im = im.astype(np.float32, copy=False)
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(target_scale) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > target_max_size:
            im_scale = float(target_max_size) / float(im_size_max)
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_processed = im_resized.transpose(2, 0, 1)
        im_processed = torch.from_numpy(im_processed).to(torch.device(cfg.DEVICE))
        ims_processed.append(im_processed)
    return ims_processed


def get_detection_params():
    # default
    nms_thresh = 0.5
    detections_per_img = 100
    # faster r-cnn  (including cascade r-cnn)
    if cfg.MODEL.FASTER_ON:
        nms_thresh = cfg.FAST_RCNN.NMS_TH
        detections_per_img = cfg.FAST_RCNN.DETECTIONS_PER_IMG
    return nms_thresh, detections_per_img
