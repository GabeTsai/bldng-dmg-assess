# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import wandb

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, wandb_run=None, 
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    if wandb_run is not None and misc.is_main_process():
        global_step = epoch * len(data_loader)
        wandb_run.log({'epoch': epoch}, step=global_step)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast("cuda"):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        # Gradient accumulation
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0) 
        if args.log_grad and misc.is_main_process():
            l2_norm = misc.global_l2_norm_grad(model.parameters())
            wandb_run.log({'train/l2_norm_grad': l2_norm}, 
                            step=global_step + data_iter_step)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (data_iter_step + 1) % accum_iter == 0:
            """ 
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)

            if log_writer is not None:
                log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', max_lr, epoch_1000x)
            
            if wandb_run is not None and misc.is_main_process():
                cur_step = global_step + data_iter_step
                wandb_run.log({
                    'train/loss': loss_value_reduce,
                    'train/lr': max_lr,
                    'epoch': epoch + data_iter_step / len(data_loader),
                    'epoch_1000x': epoch_1000x
                }, step=cur_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    final_step = None
    if wandb_run is not None and misc.is_main_process():
        final_step = data_iter_step + global_step
        avg_stats = {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
        avg_stats['epoch'] = epoch
        wandb_run.log(avg_stats, step=final_step)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, final_step

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, wandb_run=None, step=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.autocast("cuda"):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if wandb_run is not None and step is not None and misc.is_main_process():
        print("AGAGAGAGAGGAG")
        wandb_run.log({f"val/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}|
                      {"val/epoch": epoch}, step = step)
    return avg_stats
