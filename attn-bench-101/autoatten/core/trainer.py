#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/trainer.py
"""

import os
import random
import warnings
import torch.nn.functional as F
import numpy as np
import autoatten.core.benchmark as benchmark
import autoatten.core.builders as builders
import autoatten.core.checkpoint as cp
import autoatten.core.config as config
import autoatten.core.distributed as dist
import autoatten.core.logging as logging
import autoatten.core.meters as meters
import autoatten.core.net as net
import autoatten.core.optimizer as optim
import autoatten.datasets.loader as data_loader
import torch
import torch.cuda.amp as amp
from autoatten.core.config import cfg
from autoatten.core.io import cache_url, pathmgr


logger = logging.get_logger(__name__)


def setup_env():
    if dist.is_main_proc():
        pathmgr.mkdirs(cfg.OUT_DIR)
        config.dump_cfg()
    logging.setup_logging()
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    env = "".join([f"{key}: {value}\n" for key, value in sorted(os.environ.items())])
    logger.info(f"os.environ:\n{env}")
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model(setup_ema=True):
    model = builders.build_model(cfg)
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # logger.info(logging.dump_log_data(net.unwrap_model(model).complexity(), "complexity"))
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    model_state = model.state_dict()
    if cfg.NUM_GPUS > 1:
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(module=model, device_ids=[cur_device], output_device=cur_device)
    if not setup_ema:
        return model
    else:
        ema = builders.build_model(cfg)
        ema = ema.cuda(device=cur_device)
        ema.load_state_dict(model_state)
        if cfg.NUM_GPUS > 1:
            ddp = torch.nn.parallel.DistributedDataParallel
            ema = ddp(module=ema, device_ids=[cur_device], output_device=cur_device)
        return model, ema


def get_weights_file(weights_file):
    download = dist.is_main_proc(local=True)
    weights_file = cache_url(weights_file, cfg.DOWNLOAD_CACHE, download=download)
    if cfg.NUM_GPUS > 1:
        torch.distributed.barrier()
    return weights_file


def train_epoch(loader, model, ema, loss_fun, optimizer, scheduler, scaler, meter, cur_epoch):
    data_loader.shuffle(loader, cur_epoch)
    lr = optim.get_current_lr(optimizer)
    model.train()
    ema.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels, offline_features) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        offline_features = [f.cuda() for f in offline_features]
        labels_one_hot = net.smooth_one_hot_labels(labels)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            loss_cls = loss_fun(preds, labels_one_hot)
            loss, loss_inter, loss_logit = loss_cls, inputs.new_tensor(0.0), inputs.new_tensor(0.0)
            if hasattr(net.unwrap_model(model), 'guidance_loss'):
                loss_inter, loss_logit = net.unwrap_model(model).guidance_loss(inputs, offline_features)
                if cfg.DISTILLATION.ENABLE_LOGIT:
                    loss_cls = loss_cls * (1 - cfg.DISTILLATION.LOGIT_WEIGHT)
                    loss_logit = loss_logit * cfg.DISTILLATION.LOGIT_WEIGHT
                    loss = loss_cls + loss_logit
                if cfg.DISTILLATION.ENABLE_INTER:
                    loss_inter = loss_inter * cfg.DISTILLATION.INTER_WEIGHT
                    loss = loss_cls + loss_inter
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        net.update_model_ema(model, ema, cur_epoch, cur_iter)
        top1_acc, top5_acc = meters.topk_accs(preds, labels, [1, 5])
        loss_cls, loss_inter, loss_logit, loss, top1_acc, top5_acc = dist.scaled_all_reduce([loss_cls, loss_inter, loss_logit, loss, top1_acc, top5_acc])
        loss_cls, loss_inter, loss_logit, loss, top1_acc, top5_acc = loss_cls.item(), loss_inter.item(), loss_logit.item(), loss.item(), top1_acc.item(), top5_acc.item()
        meter.iter_toc()
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(top1_acc, top5_acc, loss_cls, loss_inter, loss_logit, loss, lr, mb_size)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    meter.log_epoch_stats(cur_epoch)
    scheduler.step(cur_epoch + 1)


@torch.no_grad()
def test_epoch(loader, model, meter, cur_epoch):
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels, _) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        top1_acc, top5_acc = meters.topk_accs(preds, labels, [1, 5])
        top1_acc, top5_acc = dist.scaled_all_reduce([top1_acc, top5_acc])
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()
        meter.iter_toc()
        meter.update_stats(top1_acc, top5_acc, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    meter.log_epoch_stats(cur_epoch)


def train_model():
    setup_env()
    model, ema = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    scheduler = optim.construct_scheduler(optimizer)
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cp.has_checkpoint():
        # if cfg.DISTILLATION.ENABLE_INTER and cfg.DISTILLATION.INTER_TRANSFORM == 'linear':
        #     warnings.warn('Linear transform is not supported for resuming. This will cause the linear transformation to be trained from scratch.')
        file = cp.get_last_checkpoint()
        logger.info("Loaded checkpoint from: {}".format(file))
        epoch = cp.load_checkpoint(file, model, ema, optimizer)[0]
        start_epoch = epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        train_weights = get_weights_file(cfg.TRAIN.WEIGHTS)
        logger.info("Loaded initial weights from: {}".format(train_weights))
        cp.load_checkpoint(train_weights, model, ema)
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    ema_meter = meters.TestMeter(len(test_loader), "test_ema")
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    logger.info("Start epoch: {}".format(start_epoch + 1))

    res =[]
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        params = (train_loader, model, ema, loss_fun, optimizer, scheduler, scaler, train_meter)
        train_epoch(*params, cur_epoch)
        test_epoch(test_loader, model, test_meter, cur_epoch)
        test_acc = test_meter.get_epoch_stats(cur_epoch)["top1_acc"]
        res.append(test_meter.get_epoch_stats(cur_epoch)["max_top1_acc"])
        ema_acc = 0.0
        if cfg.OPTIM.EMA_UPDATE_PERIOD > 0:
            test_epoch(test_loader, ema, ema_meter, cur_epoch)
            ema_acc = ema_meter.get_epoch_stats(cur_epoch)["top1_acc"]
        if cur_epoch % 50==0 or cur_epoch+1 == cfg.OPTIM.MAX_EPOCH:
            file = cp.save_checkpoint(model, ema, optimizer, cur_epoch, test_acc, ema_acc)
            logger.info("Wrote checkpoint to: {}".format(file))
        if (cur_epoch+1) == 100 and test_acc < 15.0:
            logger.info("Accuray {} % less than 15.0 %, exit ... ".format(test_acc))
            break
        if cur_epoch != 0 and  np.isnan(train_meter.get_epoch_stats(cur_epoch)["loss"]):
            logger.info("Loss is nan, exit ... ")
            break
        # if len(res) >= 25 and len(set(res[-25:])) == 1:
        #     logger.info("Same Max Top-1 Accuray {}, exit ... ".format(res))
        #     break




def test_model():
    setup_env()
    model = setup_model(setup_ema=False)
    test_weights = get_weights_file(cfg.TEST.WEIGHTS)
    cp.load_checkpoint(test_weights, model)
    logger.info("Loaded model weights from: {}".format(test_weights))
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    setup_env()
    model = setup_model(setup_ema=False)
    loss_fun = builders.build_loss_fun().cuda()
    benchmark.compute_time_model(model, loss_fun)


def time_model_and_loader():
    setup_env()
    model = setup_model(setup_ema=False)
    loss_fun = builders.build_loss_fun().cuda()
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)




