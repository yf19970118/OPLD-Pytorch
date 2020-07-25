import os
import json
import shutil
import numpy as np

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.data.evaluation.box_proposal_eval import BoxProposalEvaluator
from utils.data.structures.quad_bbox import quadboxes_iou
from rcnn.core.config import cfg


def evaluation(dataset, all_boxes, clean_up=True):
    pet_results = {}
    iou_types = ()
    output_folder = os.path.join(cfg.CKPT, 'test')
    if cfg.MODEL.HAS_BOX:
        iou_types = iou_types + ("bbox",)
        pet_results["bbox"] = all_boxes

    if cfg.RPN.RPN_ONLY:
        pet_eval = BoxProposalEvaluator(dataset, pet_results["bbox"])
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        if clean_up:  # clean up all the test files
            shutil.rmtree(output_folder)
        return

    for iou_type in iou_types:
        file_path = os.path.join(output_folder, iou_type + ".json")
        pet_eval = evaluate_on_coco(dataset.coco, pet_results[iou_type], file_path, iou_type)
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        # if use_dota:
        # dota_folder = os.path.join(cfg.CKPT, 'result')
        # get_dota_result(dota_folder, file_path, dataset.ann_file)
    # if clean_up:    # clean up all the test files
    #     shutil.rmtree(output_folder)
    return None


def post_processing(results, image_ids, dataset):
    box_results, ims_dets, ims_labels = prepare_box_results(results, image_ids, dataset)
    eval_results = box_results
    ims_results = [ims_dets, ims_labels]
    return eval_results, ims_results


def prepare_box_results(results, image_ids, dataset):
    box_results = []
    ims_dets = []
    ims_labels = []
    if cfg.RPN.RPN_ONLY:
        return results, None, None
    else:
        for i, result in enumerate(results):
            image_id = image_ids[i]
            original_id = dataset.id_to_img_map[image_id]
            if len(result) == 0:
                ims_dets.append(None)
                ims_labels.append(None)
                continue
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            result = result.resize((image_width, image_height))

            boxes = result.quad_bbox
            scores = result.get_field("scores")
            labels = result.get_field("labels")

            ims_dets.append(np.hstack((boxes.cpu(), scores.cpu()[:, np.newaxis])).astype(np.float32, copy=False))
            boxes = result.quad_bbox.tolist()
            scores = scores.tolist()
            labels = labels.tolist()
            ims_labels.append(labels)
            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
            box_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
    return box_results, ims_dets, ims_labels


def evaluate_on_coco(coco_gt, coco_results, json_result_file, iou_type="bbox"):
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()
    coco_eval = QuadCOCOeval(coco_gt, coco_dt, iou_type)
    return coco_eval


class QuadCOCOeval(COCOeval):
    def compute_iou_dt_gt(self, dt, gt):
        g = [g["segmentation"][0][:8] for g in gt]
        d = [d["bbox"] for d in dt]
        g = torch.as_tensor(g, dtype=torch.float32).cuda().reshape(-1, 8)
        d = torch.as_tensor(d, dtype=torch.float32).cuda().reshape(-1, 8)
        return quadboxes_iou(d, g)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0: p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"
        ious = self.compute_iou_dt_gt(dt, gt)
        return ious
