#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/core/config.py
"""

import os

from autoatten.core.io import pathmgr
from yacs.config import CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C







# ------------------------------- Common model options ------------------------------- #
_C.MODEL = CfgNode()

_C.MODEL.TYPE = "transformer"
_C.MODEL.IMG_SIZE = 224
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.LOSS_FUN = "cross_entropy"
_C.MODEL.STAGE_ATTN_TYPE = 'BBBB'
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.ATTENTION_DROP_RATE = 0.0




# -------------------------------- AutoFormer and PiT Transformer options ------------------------------- #

_C.AUTOFORMER_SEARCH_SPACE = CfgNode()

_C.AUTOFORMER_SEARCH_SPACE.HIDDEN_DIM = [192, 216, 240]
_C.AUTOFORMER_SEARCH_SPACE.MLP_RATIO = [3.5, 4.0]
_C.AUTOFORMER_SEARCH_SPACE.DEPTH = [12, 13, 14]
_C.AUTOFORMER_SEARCH_SPACE.NUM_HEADS = [3, 4]

_C.PIT_SEARCH_SPACE = CfgNode()
_C.PIT_SEARCH_SPACE.MLP_RATIO = [2, 4, 6, 8]
_C.PIT_SEARCH_SPACE.NUM_HEADS = [2, 4, 8]
_C.PIT_SEARCH_SPACE.DEPTH = [[1,2,3], [4,6,8], [2,4,6]]
_C.PIT_SEARCH_SPACE.BASE_DIM = [16, 24, 32, 40]


_C.AUTOFORMER = CfgNode()
_C.AUTOFORMER.HIDDEN_DIM = None
_C.AUTOFORMER.MLP_RATIO = None
_C.AUTOFORMER.DEPTH = None
_C.AUTOFORMER.NUM_HEADS = None

_C.PIT = CfgNode()
_C.PIT.BASE_DIM = None
_C.PIT.MLP_RATIO = None
_C.PIT.DEPTH = None
_C.PIT.NUM_HEADS = None

_C.PIT.STRIDE = 8



# -------------------------------- Swin Transformer ------------------------------- #

_C.SWIN = CfgNode()
_C.SWIN.PATCH_SIZE = 4
_C.SWIN.IN_CHANS = 3
_C.SWIN.EMBED_DIM = 96
_C.SWIN.DEPTHS = [2, 2, 6, 2]
_C.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.SWIN.WINDOW_SIZE = 7
_C.SWIN.MLP_RATIO = 4.
_C.SWIN.QKV_BIAS = True
_C.SWIN.QK_SCALE = None
_C.SWIN.KA = [7, 7, 7, 7]
_C.SWIN.DIM_REDUCTION = [4, 4, 4, 4]
_C.SWIN.STAGES = [True, True, True, True]
_C.SWIN.STAGES_NUM = [-1, -1, -1, -1]
_C.SWIN.RPB = True
_C.SWIN.PADDING_MODE = 'zeros'
_C.SWIN.SHARE_DWC_KERNEL = True
_C.SWIN.SHARE_QKV = False
_C.SWIN.APE = False
_C.SWIN.PATCH_NORM = True
_C.SWIN.LR_FACTOR = 2
_C.SWIN.DEPTHS_LR = [2, 2, 2, 2]
_C.SWIN.FUSION_TYPE = 'add'
_C.SWIN.STAGE_CFG = None





















_C.Complex = CfgNode()
_C.Complex.param = None
_C.Complex.flops = None






_C.PVT = CfgNode()

_C.PVT.LA_SR_RATIOS = 1111





# ----------------- atten-zero------------------------------------

_C.AT = CfgNode()
_C.AT.GRAPH_TYPE = None
_C.AT.UNARY_OP = None
_C.AT.BINARY_OP = None



# ----------------- Flatten Attention------------------------------------

_C.FLATTEN = CfgNode()
_C.FLATTEN.FOCUSING_FACTOR = 3
_C.FLATTEN.KERNEL_SIZE = 5




# -------------------------------- Optimizer options --------------------------------- #
_C.OPTIM = CfgNode()

# Type of optimizer select from {'sgd', 'adam', 'adamw'}
_C.OPTIM.OPTIMIZER = "adamw"

# Learning rate of body ranges from BASE_LR to MIN_LR according to the LR_POLICY
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.MIN_LR = 0.0

# Base learning of head is TRANSFER_LR_RATIO * BASE_LR
_C.OPTIM.HEAD_LR_RATIO = 1.0

# Learning rate policy select from {'cos', 'exp', 'lin', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# Betas (for Adam/AdamW optimizer)
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.BETA2 = 0.999

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Exponential Moving Average (EMA) update value
_C.OPTIM.EMA_ALPHA = 1e-5

# Iteration frequency with which to update EMA weights
_C.OPTIM.EMA_UPDATE_PERIOD = 0


# --------------------------------- Training options --------------------------------- #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

# If True train using mixed precision
_C.TRAIN.MIXED_PRECISION = False

# Label smoothing value in 0 to 1 where (0 gives no smoothing)
_C.TRAIN.LABEL_SMOOTHING = 0.0

# Batch mixup regularization value in 0 to 1 (0 gives no mixup)
_C.TRAIN.MIXUP_ALPHA = 0.0

# Batch cutmix regularization value in 0 to 1 (0 gives no cutmix)
_C.TRAIN.CUTMIX_ALPHA = 0.0

_C.TRAIN.STRONG_AUGMENTATION = True
_C.TRAIN.USE_CHECKPOINT = False


# --------------------------------- Testing options ---------------------------------- #
_C.TEST = CfgNode()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ------------------------------- Data loader options -------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------- CUDNN options ----------------------------------- #
_C.CUDNN = CfgNode()

# Perform benchmarking to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True


# ------------------------------- Precise time options ------------------------------- #
_C.PREC_TIME = CfgNode()

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------- Launch options ---------------------------------- #
_C.LAUNCH = CfgNode()

# The launch mode, may be 'local' or 'slurm' (or 'submitit_local' for debugging)
# The 'local' mode uses a multi-GPU setup via torch.multiprocessing.run_processes.
# The 'slurm' mode uses submitit to launch a job on a SLURM cluster and provides
# support for MULTI-NODE jobs (and is the only way to launch MULTI-NODE jobs).
# In 'slurm' mode, the LAUNCH options below can be used to control the SLURM options.
# Note that NUM_GPUS (not part of LAUNCH options) determines total GPUs requested.
_C.LAUNCH.MODE = "local"

# Launch options that are only used if LAUNCH.MODE is 'slurm'
_C.LAUNCH.MAX_RETRY = 3
_C.LAUNCH.NAME = "pycls_job"
_C.LAUNCH.COMMENT = ""
_C.LAUNCH.CPUS_PER_GPU = 10
_C.LAUNCH.MEM_PER_GPU = 60
_C.LAUNCH.PARTITION = "devlab"
_C.LAUNCH.GPU_TYPE = "volta"
_C.LAUNCH.TIME_LIMIT = 4200
_C.LAUNCH.EMAIL = ""


# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# If True output additional info to log
_C.VERBOSE = True

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Maximum number of GPUs available per node (unlikely to need to be changed)
_C.MAX_GPUS_PER_NODE = 8

# Output directory
_C.OUT_DIR = None

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism is still be present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port range for multi-process groups (actual port selected randomly)
_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"


# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_cfg():
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    err_str = "NUM_GPUS must be divisible by or less than MAX_GPUS_PER_NODE"
    num_gpus, max_gpus_per_node = _C.NUM_GPUS, _C.MAX_GPUS_PER_NODE
    assert num_gpus <= max_gpus_per_node or num_gpus % max_gpus_per_node == 0, err_str
    err_str = "Invalid mode {}".format(_C.LAUNCH.MODE)
    assert _C.LAUNCH.MODE in ["local", "submitit_local", "slurm"], err_str


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)
