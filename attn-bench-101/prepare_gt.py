from scipy.stats import stats
import random
import os
import subprocess
import sys
import time
import yaml
import copy
import glob
import autoatten.core.logging as logging
import torch
import time,argparse
import autoatten.core.builders as builders
import autoatten.core.config as config
from autoatten.core.config import cfg

from autoatten.models import build_model
import collections

import autoatten.core.net as net
import numpy as np
import json
from thop import profile

logger = logging.get_logger(__name__)








def obtain_trial_acc(save_path):

    data = open(save_path).readlines()[-1]
    res_dict = json.loads(data[12:])
    if res_dict['epoch'] == '300/300':
        return res_dict['max_top1_acc']
    else:
        return None



def obtain_trial_cfg(cfg_path):
    config = {}
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    config['graph_type'] = data['AT']['GRAPH_TYPE']
    config['unary_op'] = data['AT']['UNARY_OP']
    config['binary_op'] = data['AT']['BINARY_OP']
    return config






def prepare_trials(xargs):

    bash_file_set = [['#!/bin/bash'] for _ in range(xargs.gpu_num)]


    popl = torch.load(xargs.info_path)

    print('before train, pop:')
    for item in popl:
        print(item)

    config.load_cfg(xargs.refer_cfg)
    config.assert_cfg()


    for idx, cand in enumerate(popl):

        trial_name = 'Trial-{}'.format(idx)

        trial_save = xargs.save_dir + '/'  + trial_name

        if not os.path.exists(trial_save):
            os.makedirs(trial_save, exist_ok=True)


        with open(xargs.refer_cfg) as f:
            refer_data = yaml.safe_load(f)
        trial_data = copy.deepcopy(refer_data)

        trial_data['AT']['GRAPH_TYPE'] = cand['graph_type']
        trial_data['AT']['UNARY_OP'] = cand['unary_op']
        trial_data['AT']['BINARY_OP'] = cand['binary_op']

        trial_data['OPTIM']['MAX_EPOCH'] = xargs.trial_epoch
        trial_data['OUT_DIR'] = trial_save

        cfg.AT.GRAPH_TYPE = cand['graph_type']
        cfg.AT.UNARY_OP = cand['unary_op']
        cfg.AT.BINARY_OP =  cand['binary_op']

        #





        tmp_model = build_model(cfg)
        tmp_inputs = torch.randn(1, 3, 224, 224)
        flops, params = profile(tmp_model, (tmp_inputs,))
        trial_data['Complex'] ={}
        trial_data['Complex']['param'] = round(params/1e6, 2)
        trial_data['Complex']['flops'] = round(flops/1e9, 2)

        with open(trial_save+'/{}.yaml'.format(trial_name), 'w') as f:
            yaml.safe_dump(trial_data, f)


        execution_line = "CUDA_VISIBLE_DEVICES={}  python run_net.py --mode train --cfg {}/{}.yaml".format(
            int(idx % xargs.gpu_num), trial_save, trial_name )

        bash_file_set[idx % xargs.gpu_num].append(execution_line)

    for i in range(xargs.gpu_num):
        with open(os.path.join(xargs.save_dir, 'run_bash_{}.sh'.format(i)), 'w') as handle:
            for line in bash_file_set[i]:
                handle.write(line + os.linesep)




def save_trials(args):
    res = []
    for file in glob.glob(os.path.join(args.save_dir, 'Trial-*', )):
        arch_idx = file.split('-')[-1]
        print('idx:', arch_idx)
        temp = {}.fromkeys(('idx', 'arch', 'save', 'acc'))
        temp['idx'] = arch_idx
        temp['save'] = file
        cfg_path = os.path.join(file, 'Trial-{}.yaml'.format(arch_idx))
        log_path = file + '/log.txt'
        temp['arch'] = obtain_trial_cfg(cfg_path)
        temp['acc'] = obtain_trial_acc(log_path)
        res.append(temp)
    save_res= sorted(res, key=lambda x: int(x['idx']))
    print(save_res)
    torch.save(save_res, os.path.join(args.save_dir, 'trial_info.pth'))
    print('finish saving')




#
if __name__ == "__main__":
    parser = argparse.ArgumentParser("gt")

    # train
    parser.add_argument('--gpu_num', default=8, type=int, help='total gpu number')
    parser.add_argument('--trial_epoch', default=300, type=int, help='train epoch')

    parser.add_argument('--save_dir', default=None, type=str)

    parser.add_argument('--refer_cfg', default='./configs/autoformer-linear_c100_base.yaml', type=str, help='save output path')
    parser.add_argument('--info_path', default=None, type=str,
                        help='load population from a saved info path')
    parser.add_argument('--mode', default='prepare', type=str, help='prepare, save')

    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(k,' = ',v)
    if not args.save_dir:
        args.save_dir = os.path.join('work_dirs/GT/', os.path.splitext(os.path.basename(args.refer_cfg))[0])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == 'prepare':
        prepare_trials(args)
    elif args.mode == 'save':
        save_trials(args)