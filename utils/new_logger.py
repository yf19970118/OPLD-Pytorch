import os
import time
import shutil
import logging
import datetime
import numpy as np

import torch

from utils.timer import Timer
from utils.events import EventStorage, EventWriter, CommonMetricPrinter, JSONWriter, TensorboardXWriter
from utils.misc import logging_rank, is_main_process, gather


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


class TrainHook:
    def before_train(self, **kwargs):
        pass

    def after_train(self, **kwargs):
        pass

    def before_step(self, **kwargs):
        pass

    def after_step(self, **kwargs):
        pass


class TestHook():
    """Track vital testing statistics."""

    def __init__(self, cfg_filename, log_period=10):
        self.cfg_filename = cfg_filename
        self.log_period = log_period

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.infer_timer = Timer()
        self.post_timer = Timer()

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        return self.iter_timer.toc(average=False)

    def data_tic(self):
        self.data_timer.tic()

    def data_toc(self):
        return self.data_timer.toc(average=False)

    def infer_tic(self):
        self.infer_timer.tic()

    def infer_toc(self):
        return self.infer_timer.toc(average=False)

    def post_tic(self):
        self.post_timer.tic()

    def post_toc(self):
        return self.post_timer.toc(average=False)

    def reset_timer(self):
        self.iter_timer.reset()
        self.data_timer.reset()
        self.infer_timer.reset()
        self.post_timer.reset()

    def log_stats(self, cur_idx, start_ind, end_ind, total_num_images, suffix=None):
        """Log the tracked statistics."""
        if cur_idx % self.log_period == 0:
            eta_seconds = self.iter_timer.average_time * (end_ind - cur_idx - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            lines = '[Testing][range:{}-{} of {}][{}/{}]'. \
                format(start_ind + 1, end_ind, total_num_images, cur_idx + 1, end_ind)

            lines += '[{:.3f}s = {:.3f}s + {:.3f}s + {:.3f}s][eta: {}]'. \
                format(self.iter_timer.average_time, self.data_timer.average_time, self.infer_timer.average_time,
                       self.post_timer.average_time, eta)
            if suffix is not None:
                lines += suffix
            logging_rank(lines)
        return None


class PeriodicWriter(TrainHook):
    """
    Write events to EventStorage periodically.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, cfg, writers, max_iter):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self.cfg = cfg
        self._writers = writers
        self._max_iter = max_iter
        for w in writers:
            assert isinstance(w, EventWriter), w

    def after_step(self, storage, epoch=None, **kwargs):
        if epoch is not None:
            max_epoch = self.cfg.SOLVER.MAX_EPOCHS
            iter = storage.iter % self._max_iter
        else:
            max_epoch = None
            iter = storage.iter

        if epoch is not None:
            storage.put_scalar("epoch", epoch, smoothing_hint=False)
        if (iter + 1) % self.cfg.DISPLAY_ITER == 0 or (
                iter == self._max_iter - 1
        ):
            for writer in self._writers:
                writer.write(epoch=epoch, max_epoch=max_epoch)

    def after_train(self, **kwargs):
        for writer in self._writers:
            writer.close()


class IterationTimer(TrainHook):
    def __init__(self, max_iter, start_iter, warmup_iter, ignore_warmup_time):
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_iter = start_iter
        self._max_iter = max_iter
        self._ignore_warmup_time = ignore_warmup_time

    def before_train(self, **kwargs):
        self._start_time = time.perf_counter()
        self._total_timer = Timer()
        self._total_timer.pause()

    def after_train(self, storage, **kwargs):
        iter = storage.iter
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = iter + 1 - self._start_iter - self._warmup_iter

        if is_main_process():
            if num_iter > 0 and total_time_minus_hooks > 0:
                # Speed is meaningful only after warmup
                # NOTE this format is parsed by grep in some scripts
                logger.info(
                    "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                        num_iter,
                        str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                        total_time_minus_hooks / num_iter,
                    )
                )

            logger.info(
                "Total training time: {} ({} on hooks)".format(
                    str(datetime.timedelta(seconds=int(total_time))),
                    str(datetime.timedelta(seconds=int(hook_time))),
                )
            )

    def before_step(self, **kwargs):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self, storage, epoch=None, **kwargs):
        iter = storage.iter if epoch is None else storage.iter % self._max_iter
        # +1 because we're in after_step
        if self._ignore_warmup_time:
            # ignore warm up time cost
            if iter >= self._warmup_iter:
                sec = self._step_timer.seconds()
                storage.put_scalars(time=sec)
            else:
                self._start_time = time.perf_counter()
                self._total_timer.reset()
        else:
            sec = self._step_timer.seconds()
            storage.put_scalars(time=sec)

        self._total_timer.pause()


class LRScheduler(TrainHook):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def before_step(self, storage, **kwargs):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        storage.put_scalar("lr", lr, smoothing_hint=False)
        self._scheduler.step()


def build_train_hooks(cfg, optimizer, scheduler, max_iter, warmup_iter, ignore_warmup_time=False):
    """
    Build a list of default hooks.
    """
    start_iter = scheduler.iteration

    ret = [
        IterationTimer(max_iter, start_iter, warmup_iter, ignore_warmup_time),
        LRScheduler(optimizer, scheduler),
    ]

    if is_main_process():
        write_ret = [CommonMetricPrinter(cfg.CKPT, max_iter)]

        if cfg.TRAIN.SAVE_AS_JSON:
            write_ret.append(JSONWriter(os.path.join(cfg.CKPT, "metrics.json")))
        if cfg.TRAIN.USE_TENSORBOARD:
            log_dir = os.path.join(cfg.CKPT, "tensorboard_log")
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.mkdir(log_dir)
            write_ret.append(TensorboardXWriter(log_dir))
        ret.append(PeriodicWriter(cfg, write_ret, max_iter))

    return ret


def build_test_hooks(cfg_filename, log_period):
    return TestHook(cfg_filename, log_period)


def write_metrics(metrics_dict, storage):
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    # gather metrics among all workers for logging
    all_metrics_dict = gather(metrics_dict)

    if is_main_process():
        max_keys = ("data_time", "best_acc1")
        for m_k in max_keys:
            if m_k in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                m_v = np.max([x.pop(m_k) for x in all_metrics_dict])
                storage.put_scalar(m_k, m_v)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = sum(v if 'loss' in k else 0 for k, v in metrics_dict.items())

        storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) >= 1:
            storage.put_scalars(**metrics_dict)
