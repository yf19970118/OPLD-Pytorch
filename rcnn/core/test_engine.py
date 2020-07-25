import os
import cv2
import numpy as np

import torch

from utils.misc import logging_rank, save_object
from utils.checkpointer import get_weights, load_weights
from utils.net import convert_bn2affine_model
from utils.new_logger import build_test_hooks
from utils.timer import Timer
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils

from rcnn.core.config import cfg
from rcnn.modeling.model_builder import Generalized_RCNN
from rcnn.datasets import build_dataset, evaluation, post_processing
import rcnn.core.test as rcnn_test


def run_inference(args, ind_range=None, multi_gpu_testing=False):
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            return test_net_on_dataset(args, multi_gpu=multi_gpu_testing)
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            return test_net(args, ind_range=ind_range)

    all_results = result_getter()
    return all_results


def test_net_on_dataset(args, multi_gpu=False):
    """Run inference on a dataset."""
    dataset = build_dataset(cfg.TEST.DATASETS, is_train=False)

    total_timer = Timer()
    total_timer.tic()
    if multi_gpu:
        num_images = len(dataset)
        all_boxes = multi_gpu_test_net_on_dataset(args, num_images)
    else:
        all_boxes = test_net(args)

    total_timer.toc(average=False)
    logging_rank('Total inference time: {:.3f}s'.format(total_timer.average_time), local_rank=0)

    return evaluation(dataset, all_boxes, cfg.CLEAN_UP)


def multi_gpu_test_net_on_dataset(args, num_images):
    """Multi-gpu inference on a dataset."""
    binary_dir = os.getcwd()
    binary = os.path.join(binary_dir, args.test_net_file + '.py')
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel('detection', num_images, binary, cfg, cfg.CKPT)

    # Collate the results from each subprocess
    all_boxes = []

    for ins_data in outputs:
        all_boxes += ins_data['all_boxes']

    det_file = os.path.join(cfg.CKPT, 'test', 'detections.pkl')
    save_object(dict(all_boxes=all_boxes,), det_file)

    logging_rank('Wrote detections to: {}'.format(os.path.abspath(det_file)), local_rank=0)
    return all_boxes


def test_net(args, ind_range=None):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    dataset = build_dataset(cfg.TEST.DATASETS, is_train=False)
    all_hooks = build_test_hooks(args.cfg_file.split('/')[-1], log_period=int(np.ceil(10 / cfg.TEST.IMS_PER_GPU)))
    if ind_range is not None:
        start_ind, end_ind = ind_range
    else:
        start_ind = 0
        end_ind = len(dataset)
    model = initialize_model_from_cfg()
    all_boxes = test(model, dataset, start_ind, end_ind, all_hooks)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(cfg.CKPT, 'test', det_name)
    save_object(dict(all_boxes=all_boxes,), det_file)

    logging_rank('Wrote detections to: {}'.format(os.path.abspath(det_file)), local_rank=0)
    return all_boxes,


def test(model, dataset, start_ind, end_ind, all_hooks):
    total_num_images = len(dataset)
    all_boxes = []

    num_img = cfg.TEST.IMS_PER_GPU
    with torch.no_grad():
        for i in range(start_ind, end_ind, num_img):
            all_hooks.iter_tic()

            all_hooks.data_tic()
            ims = []
            image_ids = []
            for j in range(i, i + num_img):
                if j == end_ind:
                    break
                ims.append(dataset.pull_image(j))
                image_ids.append(j)
            all_hooks.data_toc()

            all_hooks.infer_tic()
            result, features = rcnn_test.im_detect_bbox(model, ims)
            if cfg.MODEL.OPLD_ON:
                result = rcnn_test.im_detect_opld(model, result, features)
            all_hooks.infer_toc()

            all_hooks.post_tic()
            eval_results, ims_results = post_processing(result, image_ids, dataset)
            box_results = eval_results
            ims_dets, ims_labels = ims_results
            if cfg.VIS.ENABLED:
                for k, im in enumerate(ims):
                    if len(ims_dets) == 0:
                        break
                    im_name = dataset.get_img_info(image_ids[k])['file_name']
                    vis_im = vis_utils.vis_one_image_opencv(
                        im,
                        cfg.VIS,
                        boxes=ims_dets[k],
                        classes=ims_labels[k],
                        dataset=dataset
                    )
                    cv2.imwrite(os.path.join(cfg.CKPT, 'vis', '{}'.format(im_name)), vis_im)
            all_boxes += box_results
            all_hooks.post_toc()
            all_hooks.iter_toc()
            all_hooks.log_stats(i, start_ind, end_ind, total_num_images)

    return all_boxes


def initialize_model_from_cfg():
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = Generalized_RCNN(is_train=False)
    # Load trained model
    cfg.TEST.WEIGHTS = get_weights(cfg.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, cfg.TEST.WEIGHTS)
    if cfg.MODEL.BATCH_NORM == 'freeze':
        model = convert_bn2affine_model(model)
    model.eval()
    model.to(torch.device(cfg.DEVICE))

    return model
