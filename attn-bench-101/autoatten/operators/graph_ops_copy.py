import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import random
from . import binary_operation, unary_operation
#


available_graph_candidates = []
_graph_candidates_impls = {}




def graph_candidates(name, **impl_args):

    def make_impl(func):
        def graph_candidates_impl(x, atten_cfg, *args, **kwargs):
            ret = func(x, atten_cfg, *args, **kwargs, **impl_args)
            return ret

        global _graph_candidates_impls
        if name in _graph_candidates_impls:
            raise KeyError(f'Duplicated graph_candidates! {name}')
        available_graph_candidates.append(name)
        _graph_candidates_impls[name] = graph_candidates_impl
        return func

    return make_impl




def get_graph_candidates(name, *args, **kwargs):
    results = _graph_candidates_impls[name](*args, **kwargs)
    return results





def get_graph_function(name):
    return _graph_candidates_impls[name]




@graph_candidates('i3n2_b0')
def build_g1(x, atten_cfg):

    states1, states2, states3 = [x[i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']

    A1, A2, A3= states1.clone(), states2.clone(), states3.clone()

    for u_op in unary_ops[0]:
        A1 = unary_operation(A1, u_op)
    for u_op in unary_ops[1]:
        A2 = unary_operation(A2, u_op)
    B1 = binary_operation(A1, A2, binary_ops[0])
    for u_op in unary_ops[2]:
        A3 = unary_operation(A3, u_op)
    for u_op in unary_ops[3]:
        B1 = unary_operation(B1, u_op)
    B2 = binary_operation(B1, A3, binary_ops[1])

    return B2





@graph_candidates('i3n2_b1')
def build_g2(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    A1, A2, A3 = states1.clone(), states2.clone(), states3.clone()
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    bias_index = atten_cfg['bias_index'][0]

    if bias_index == 0:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0]) + x[1]
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1])

    else:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0])
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1]) + x[1]

    return B2




@graph_candidates('i3n2_b2')
def build_g3(x, atten_cfg):

    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']

    A1, A2, A3 = states1.clone(), states2.clone(), states3.clone()

    for u_op in unary_ops[0]:
        A1 = unary_operation(A1, u_op)
    for u_op in unary_ops[1]:
        A2 = unary_operation(A2, u_op)
    B1 = binary_operation(A1, A2, binary_ops[0]) + x[1]
    for u_op in unary_ops[2]:
        A3 = unary_operation(A3, u_op)
    for u_op in unary_ops[3]:
        B1 = unary_operation(B1, u_op)
    B2 = binary_operation(B1, A3, binary_ops[1]) + x[2]

    return B2



@graph_candidates('i3n301_b0')
def build_g4(x, atten_cfg):
    states1, states2, states3 = [x[i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']

    A1, A2, A3 = states1.clone(), states2.clone(), states3.clone()
    for u_op in unary_ops[0]:
        A1 = unary_operation(A1, u_op)
    for u_op in unary_ops[1]:
        A2 = unary_operation(A2, u_op)
    B1 = binary_operation(A1, A2, binary_ops[0])
    A4 = B1
    for u_op in unary_ops[2]:
        A3 = unary_operation(A3, u_op)
    for u_op in unary_ops[3]:
        B1 = unary_operation(B1, u_op)
    B2 = binary_operation(B1, A3, binary_ops[1])
    for u_op in unary_ops[4]:
        A4 = unary_operation(A4, u_op)
    for u_op in unary_ops[5]:
        B2 = unary_operation(B2, u_op)
    B3 = binary_operation(B2, A4, binary_ops[2])
    return B3





@graph_candidates('i3n301_b1')
def build_g5(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3 = states1.clone(), states2.clone(), states3.clone()
    bias_index = atten_cfg['bias_index'][0]

    if bias_index == 0:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0]) + x[1]
        A4 = B1
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1])
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B2, A4, binary_ops[2])

    elif bias_index == 1:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0])
        A4 = B1
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1]) + x[1]
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B2, A4, binary_ops[2])

    else:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0])
        A4 = B1
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1])
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B2, A4, binary_ops[2])  + x[1]

    return B3







@graph_candidates('i3n301_b2')
def build_g6(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3 = states1.clone(), states2.clone(), states3.clone()
    bias_index = atten_cfg['bias_index']

    if bias_index == [0,1]:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0]) + x[1]
        A4 = B1
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1]) + x[2]
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B2, A4, binary_ops[2])

    elif bias_index == [0,2]:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0]) + x[1]
        A4 = B1
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1])
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B2, A4, binary_ops[2])  + x[2]

    else:
        for u_op in unary_ops[0]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[1]:
            A2 = unary_operation(A2, u_op)
        B1 = binary_operation(A1, A2, binary_ops[0])
        A4 = B1
        for u_op in unary_ops[2]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(B1, A3, binary_ops[1]) + x[1]
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B2, A4, binary_ops[2])  + x[2]

    return B3






@graph_candidates('i3n302_b0')
def build_g7(x, atten_cfg):
    states1, states2, states3 = [x[i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']

    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    for u_op in unary_ops[0]:
        A2 = unary_operation(A2, u_op)
    for u_op in unary_ops[1]:
        A3 = unary_operation(A3, u_op)
    for u_op in unary_ops[2]:
        A4 = unary_operation(A4, u_op)
    B1 = binary_operation(A2, A3, binary_ops[0])
    for u_op in unary_ops[3]:
        B1 = unary_operation(B1, u_op)
    B2 = binary_operation(A4, B1, binary_ops[1])
    for u_op in unary_ops[4]:
        A1 = unary_operation(A1, u_op)
    for u_op in unary_ops[5]:
        B2 = unary_operation(B2, u_op)
    B3 = binary_operation(A1, B2, binary_ops[2])
    return B3






@graph_candidates('i3n302_b1')
def build_g8(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    bias_index = atten_cfg['bias_index'][0]
    if bias_index == 0:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[2]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(A4, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2])

    elif bias_index == 1:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[2]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(A4, B1, binary_ops[1])+ x[1]
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2])

    else:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[2]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(A4, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2]) + x[1]

    return B3



@graph_candidates('i3n302_b2')
def build_g9(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    bias_index = atten_cfg['bias_index']
    if bias_index == [0,1]:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[2]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(A4, B1, binary_ops[1]) + x[2]
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2])

    elif bias_index == [0,2]:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[2]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(A4, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2]) + x[2]

    else:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[2]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        B2 = binary_operation(A4, B1, binary_ops[1]) + x[1]
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2]) + x[2]

    return B3




@graph_candidates('i3n303_b0')
def build_g10(x, atten_cfg):
    states1, states2, states3 = [x[i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states1.clone(), states2.clone(), states3.clone()
    for u_op in unary_ops[0]:
        A3 = unary_operation(A3, u_op)
    for u_op in unary_ops[1]:
        A4 = unary_operation(A4, u_op)
    B1 = binary_operation(A3, A4, binary_ops[0])
    for u_op in unary_ops[3]:
        B1 = unary_operation(B1, u_op)
    for u_op in unary_ops[2]:
        A2 = unary_operation(A2, u_op)
    B2 = binary_operation(A2, B1, binary_ops[1])
    for u_op in unary_ops[4]:
        A1 = unary_operation(A1, u_op)
    for u_op in unary_ops[5]:
        B2 = unary_operation(B2, u_op)
    B3 = binary_operation(A1, B2, binary_ops[2])
    return B3



@graph_candidates('i3n303_b1')
def build_g11(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states1.clone(), states2.clone(), states3.clone()
    bias_index = atten_cfg['bias_index'][0]

    if bias_index == 0:
        for u_op in unary_ops[0]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[1]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A3, A4, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A2 = unary_operation(A2, u_op)
        B2 = binary_operation(A2, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2])

    elif bias_index ==1:
        for u_op in unary_ops[0]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[1]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A3, A4, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A2 = unary_operation(A2, u_op)
        # print('B2 before add:', binary_operation(A2, B1, binary_ops[1]).size())
        # print('x[1] bias before add:', x[1].size())
        B2 = binary_operation(A2, B1, binary_ops[1])  + x[1]
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2])

    else:
        for u_op in unary_ops[0]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[1]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A3, A4, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A2 = unary_operation(A2, u_op)
        B2 = binary_operation(A2, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        # print('B3 before add:', binary_operation(A1, B2, binary_ops[2]).size())
        # print('bias before add:', x[1].size())
        B3 = binary_operation(A1, B2, binary_ops[2])  + x[1]
    return B3



@graph_candidates('i3n303_b2')
def build_g12(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states1.clone(), states2.clone(), states3.clone()
    bias_index = atten_cfg['bias_index']

    if bias_index == [0, 1]:
        for u_op in unary_ops[0]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[1]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A3, A4, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A2 = unary_operation(A2, u_op)
        B2 = binary_operation(A2, B1, binary_ops[1]) + x[2]
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2])

    elif bias_index == [0,2]:
        for u_op in unary_ops[0]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[1]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A3, A4, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A2 = unary_operation(A2, u_op)
        B2 = binary_operation(A2, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2]) + x[2]

    else:
        for u_op in unary_ops[0]:
            A3 = unary_operation(A3, u_op)
        for u_op in unary_ops[1]:
            A4 = unary_operation(A4, u_op)
        B1 = binary_operation(A3, A4, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A2 = unary_operation(A2, u_op)
        B2 = binary_operation(A2, B1, binary_ops[1]) + x[1]
        for u_op in unary_ops[4]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A1, B2, binary_ops[2]) + x[2]
    return B3






@graph_candidates('i3n304_b0')
def build_g13(x, atten_cfg):
    states1, states2, states3 = [x[i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    for u_op in unary_ops[0]:
        A2 = unary_operation(A2, u_op)
    for u_op in unary_ops[1]:
        A3 = unary_operation(A3, u_op)
    B1 = binary_operation(A2, A3, binary_ops[0])
    for u_op in unary_ops[3]:
        B1 = unary_operation(B1, u_op)
    for u_op in unary_ops[2]:
        A1 = unary_operation(A1, u_op)
    B2 = binary_operation(A1, B1, binary_ops[1])
    for u_op in unary_ops[4]:
        A4 = unary_operation(A4, u_op)
    for u_op in unary_ops[5]:
        B2 = unary_operation(B2, u_op)
    B3 = binary_operation(A4, B2, binary_ops[2])
    return B3



@graph_candidates('i3n304_b1')
def build_g14(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    bias_index = atten_cfg['bias_index'][0]

    if bias_index == 0:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        B2 = binary_operation(A1, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A4, B2, binary_ops[2])
    elif bias_index == 1:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        B2 = binary_operation(A1, B1, binary_ops[1]) + x[1]
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A4, B2, binary_ops[2])
    else:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        B2 = binary_operation(A1, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A4, B2, binary_ops[2]) + x[1]
    return B3


@graph_candidates('i3n304_b2')
def build_g15(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    bias_index = atten_cfg['bias_index']

    if bias_index == [0,1]:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        B2 = binary_operation(A1, B1, binary_ops[1]) + x[2]
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A4, B2, binary_ops[2])
    elif bias_index == [0, 2]:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        B2 = binary_operation(A1, B1, binary_ops[1])
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A4, B2, binary_ops[2]) + x[2]
    else:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])
        for u_op in unary_ops[3]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        B2 = binary_operation(A1, B1, binary_ops[1]) + x[1]
        for u_op in unary_ops[4]:
            A4 = unary_operation(A4, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(A4, B2, binary_ops[2]) + x[2]
    return B3





@graph_candidates('i3n305_b0')
def build_g16(x, atten_cfg):
    states1, states2, states3 = [x[i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    for u_op in unary_ops[0]:
        A2 = unary_operation(A2, u_op)
    for u_op in unary_ops[1]:
        A3 = unary_operation(A3, u_op)
    B1 = binary_operation(A2, A3, binary_ops[0])

    for u_op in unary_ops[2]:
        A1 = unary_operation(A1, u_op)
    for u_op in unary_ops[3]:
        A4 = unary_operation(A4, u_op)
    B2 = binary_operation(A1, A4, binary_ops[1])

    for u_op in unary_ops[4]:
        B1 = unary_operation(B1, u_op)
    for u_op in unary_ops[5]:
        B2 = unary_operation(B2, u_op)
    B3 = binary_operation(B1, B2, binary_ops[2])
    return B3



@graph_candidates('i3n305_b1')
def build_g17(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    bias_index = atten_cfg['bias_index'][0]

    if bias_index ==0:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]

        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[3]:
            A4 = unary_operation(A4, u_op)
        B2 = binary_operation(A1, A4, binary_ops[1])

        for u_op in unary_ops[4]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B1, B2, binary_ops[2])

    elif bias_index ==1:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])

        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[3]:
            A4 = unary_operation(A4, u_op)
        B2 = binary_operation(A1, A4, binary_ops[1]) + x[1]

        for u_op in unary_ops[4]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B1, B2, binary_ops[2])

    else:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])

        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[3]:
            A4 = unary_operation(A4, u_op)
        B2 = binary_operation(A1, A4, binary_ops[1])

        for u_op in unary_ops[4]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B1, B2, binary_ops[2]) + x[1]
    return B3


@graph_candidates('i3n305_b2')
def build_g18(x, atten_cfg):
    states1, states2, states3 = [x[0][i] for i in range(3)]
    binary_ops = atten_cfg['binary_op']
    unary_ops = atten_cfg['unary_op']
    A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
    bias_index = atten_cfg['bias_index']

    if bias_index == [0,1]:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]

        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[3]:
            A4 = unary_operation(A4, u_op)
        B2 = binary_operation(A1, A4, binary_ops[1]) + x[2]

        for u_op in unary_ops[4]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B1, B2, binary_ops[2])

    elif bias_index == [0, 2]:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0]) + x[1]

        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[3]:
            A4 = unary_operation(A4, u_op)
        B2 = binary_operation(A1, A4, binary_ops[1])

        for u_op in unary_ops[4]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B1, B2, binary_ops[2]) + x[2]

    else:
        for u_op in unary_ops[0]:
            A2 = unary_operation(A2, u_op)
        for u_op in unary_ops[1]:
            A3 = unary_operation(A3, u_op)
        B1 = binary_operation(A2, A3, binary_ops[0])

        for u_op in unary_ops[2]:
            A1 = unary_operation(A1, u_op)
        for u_op in unary_ops[3]:
            A4 = unary_operation(A4, u_op)
        B2 = binary_operation(A1, A4, binary_ops[1]) + x[1]

        for u_op in unary_ops[4]:
            B1 = unary_operation(B1, u_op)
        for u_op in unary_ops[5]:
            B2 = unary_operation(B2, u_op)
        B3 = binary_operation(B1, B2, binary_ops[2]) + x[2]
    return B3

# @graph_candidates('i3n401')
# def build_g7(x, atten_cfg):
#     q, k, v = [x[i] for i in range(3)]
#     binary_ops = atten_cfg['binary_op']
#     unary_ops = atten_cfg['unary_op']
#     A1, A2, A3, A4, A5 = q.clone(), q.clone(), k.clone(), k.clone(), v.clone()
#
#     for u_op in unary_ops[0]:
#         A4 = unary_operation(A4, u_op)
#     for u_op in unary_ops[1]:
#         A5 = unary_operation(A5, u_op)
#     B1 = binary_operation(A4, A5, binary_ops[0])
#
#     for u_op in unary_ops[2]:
#         A2 = unary_operation(A2, u_op)
#     for u_op in unary_ops[3]:
#         A3 = unary_operation(A3, u_op)
#     B2 = binary_operation(A2, A3, binary_ops[1])
#
#     for u_op in unary_ops[4]:
#         A1 = unary_operation(A1, u_op)
#     for u_op in unary_ops[5]:
#         B1 = unary_operation(B1, u_op)
#     B3 = binary_operation(A1, B1, binary_ops[2])
#
#     for u_op in unary_ops[6]:
#         B2 = unary_operation(B2, u_op)
#     for u_op in unary_ops[7]:
#         B3 = unary_operation(B3, u_op)
#     B4 = binary_operation(B2, B3, binary_ops[3])
#     return B4
#
#
#
#
#
# @graph_candidates('i3n402')
# def build_g8(x, atten_cfg):
#     states1, states2, states3 = [x[i] for i in range(3)]
#     binary_ops = atten_cfg['binary_op']
#     unary_ops = atten_cfg['unary_op']
#     A1, A2, A3, A4 = states1.clone(), states2.clone(), states3.clone(), states3.clone()
#
#     for u_op in unary_ops[0]:
#         A2 = unary_operation(A2, u_op)
#     for u_op in unary_ops[1]:
#         A3 = unary_operation(A3, u_op)
#     B1 = binary_operation(A2, A3, binary_ops[0])
#
#     for u_op in unary_ops[2]:
#         A1 = unary_operation(A1, u_op)
#     for u_op in unary_ops[3]:
#         B1 = unary_operation(B1, u_op)
#     B2 = binary_operation(A1, B1, binary_ops[1])
#
#     for u_op in unary_ops[4]:
#         A4 = unary_operation(A4, u_op)
#     for u_op in unary_ops[5]:
#         B2 = unary_operation(B2, u_op)
#     B3 = binary_operation(A4, B2, binary_ops[2])
#
#     for u_op in unary_ops[6]:
#         B2 = unary_operation(B2, u_op)
#     for u_op in unary_ops[7]:
#         B3 = unary_operation(B3, u_op)
#     B4 = binary_operation(B2, B3, binary_ops[3])
#     return B4







