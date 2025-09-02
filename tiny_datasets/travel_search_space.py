
import autoatten.core.config as config
from autoatten.models.build import MODEL
import autoatten.core.logging as logging
from autoatten.core.config import cfg
cfg_path = 'configs/autoformer-linear_c100_base.yaml'
config.load_cfg(cfg_path)
config.assert_cfg()
import time
import os
from autoatten.operators import *

import random

import torch
#
#
logger = logging.get_logger(__name__)
logging.setup_logging()
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))


log_file = '{}.txt'.format(time_str)
file_handler = logging.FileHandler(log_file, 'w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)




def generate_att_cfg(num=1):
    res= []
    logger.info(f'Begin sample attention config...')
    #  ['att1_i3', 'att2_i4']
    for i in range(num):
        flag = False
        while not flag:
            genotype_keys = ('graph_type', 'unary_op', 'binary_op')
            tmp_cfg = dict.fromkeys(genotype_keys)
            unary_ops, binary_ops = [], []

            total_num_graphs = len(available_graph_candidates)
            graph_idx = random.choice(range(total_num_graphs))
            graph_type = available_graph_candidates[graph_idx]
            tmp_cfg['graph_type'] = graph_type
            binary_num = int(graph_type[3])

            for j in range(2*binary_num):
                unary_num = random.choice([1,2,3])
                unary_ops.append([sample_unary_key_by_prob() for _ in range(unary_num)])
            for k in range(binary_num):
                binary_ops.append(sample_binary_key_by_prob())

            tmp_cfg['unary_op'] = unary_ops
            tmp_cfg['binary_op'] = binary_ops

            try:
                model = MODEL.get(cfg.MODEL.TYPE)(tmp_cfg)
                inputs = torch.randn(200, 3, 224, 224)
                model(inputs)
                logger.info(f'Valid config: {tmp_cfg}')
            except Exception as e:
                logger.info(f'Invalid  config: {tmp_cfg}')
                # logger.info(e)
                continue
            if tmp_cfg not in res:
                res.append(tmp_cfg)
                flag = True

    return res



ress = generate_att_cfg(10)

for item in ress:
    logger.info(item)

torch.save(ress,'att_candidates.pth')