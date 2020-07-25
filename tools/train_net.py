import os
import time
import shutil
import argparse

import torch
import torch.nn as nn

import _init_paths  # pylint: disable=unused-import
from utils.misc import mkdir_p, logging_rank
from utils.net import convert_bn2affine_model, convert_bn2syncbn_model, mismatch_params_filter
from utils.checkpointer import CheckPointer
from utils.optimizer import Optimizer
from utils.lr_scheduler import LearningRateScheduler
from utils.events import EventStorage
from utils.new_logger import build_train_hooks, write_metrics

from rcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from rcnn.datasets import build_dataset, make_train_data_loader
from rcnn.modeling.model_builder import Generalized_RCNN

# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('opts', help='See pet/rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)

args.device = torch.device(cfg.DEVICE)
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    args.world_size = torch.distributed.get_world_size()
else:
    args.world_size = 1
    args.local_rank = 0
    cfg.NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if cfg.DEVICE == 'cuda' else 1
    cfg.TRAIN.LOADER_THREADS *= cfg.NUM_GPUS
    cfg.TEST.LOADER_THREADS *= cfg.NUM_GPUS
    cfg.TEST.IMS_PER_GPU *= cfg.NUM_GPUS

logging_rank('Called with args: {}'.format(args), distributed=args.distributed, local_rank=args.local_rank)


def train(model, loader, optimizer, scheduler, checkpointer, all_hooks):
    # switch to train mode
    model.train()

    # main loop
    start_iter = scheduler.iteration
    loader = iter(enumerate(loader, start_iter))
    logging_rank("Starting training from iteration {}".format(start_iter), distributed=args.distributed,
                 local_rank=args.local_rank)

    with EventStorage(start_iter=start_iter, log_period=cfg.DISPLAY_ITER) as storage:
        try:
            for h in all_hooks:
                h.before_train()
            for iteration in range(start_iter, cfg.SOLVER.MAX_ITER):
                for h in all_hooks:
                    h.before_step(storage=storage)

                data_start = time.perf_counter()
                _, (images, targets, _) = next(loader)
                images = images.to(args.device)
                targets = [target.to(args.device) for target in targets]
                data_time = time.perf_counter() - data_start

                optimizer.zero_grad()

                outputs = model(images, targets)
                losses = sum(loss for loss in outputs['losses'].values())
                metrics_dict = outputs['losses']
                metrics_dict["data_time"] = data_time
                write_metrics(metrics_dict, storage)
                losses.backward()
                optimizer.step()

                if args.local_rank == 0:
                    # Save model
                    if cfg.SOLVER.SNAPSHOT_ITERS > 0 and (iteration + 1) % cfg.SOLVER.SNAPSHOT_ITERS == 0:
                        checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix='iter')
                for h in all_hooks:
                    h.after_step(storage=storage)
                storage.step()

            if args.local_rank == 0:
                checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix='iter')
        finally:
            for h in all_hooks:
                h.after_train(storage=storage)
    return None


def main():
    if not os.path.isdir(cfg.CKPT):
        mkdir_p(cfg.CKPT)
    if args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))
    assert_and_infer_cfg(make_immutable=False)

    # Create model
    model = Generalized_RCNN()
    logging_rank(model, distributed=args.distributed, local_rank=args.local_rank)

    # Create checkpointer
    checkpointer = CheckPointer(cfg.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME,
                                local_rank=args.local_rank)

    # Load model or random-initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MODEL.CONV1_RGB2BGR)
    if cfg.MODEL.BATCH_NORM == 'freeze':
        model = convert_bn2affine_model(model, merge=not checkpointer.resume)
    elif cfg.MODEL.BATCH_NORM == 'sync':
        model = convert_bn2syncbn_model(model)
    model.to(args.device)

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER, local_rank=args.local_rank).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank('The mismatch keys: {}'.format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))),
                 distributed=args.distributed, local_rank=args.local_rank)

    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, local_rank=args.local_rank)
    scheduler = checkpointer.load_scheduler(scheduler)

    # Create training dataset and loader
    datasets = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)
    train_loader = make_train_data_loader(datasets, is_distributed=args.distributed, start_iter=scheduler.iteration)

    # Model Distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
        )
    else:
        model = torch.nn.DataParallel(model)

    # Build hooks
    all_hooks = build_train_hooks(cfg, optimizer, scheduler, max_iter=cfg.SOLVER.MAX_ITER,
                                  warmup_iter=cfg.SOLVER.WARM_UP_ITERS, ignore_warmup_time=False)

    # Train
    train(model, train_loader, optimizer, scheduler, checkpointer, all_hooks)


if __name__ == '__main__':
    main()
