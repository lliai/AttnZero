# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import torch
import torch.nn as nn


def get_some_data(train_dataloader, num_batches, device):
    traindata = []
    dataloader_iter = iter(train_dataloader)
    for _ in range(num_batches):
        traindata.append(next(dataloader_iter))

    # adapt to the dataloader of pycls
    inputs = torch.cat([a for a, _, _ in traindata])
    targets = torch.cat([b for _, b, _ in traindata])
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets


def get_some_data_grasp(train_dataloader, num_classes, samples_per_class,
                        device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    while True:
        inputs, targets, _ = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device)
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y


def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric_array.append(metric(layer))

    return metric_array



# def get_layer_metric_array_dss(net, metric, mode):
#     metric_array = []
#
#     for layer in net.modules():
#         if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
#             continue
#         if isinstance(layer, nn.Linear):
#             metric_array.append(metric(layer))
#     return metric_array




def get_layer_metric_array_dss(net):
    metric_array = []

    for name, module in net.named_modules():
        if (isinstance(module, nn.Linear) and 'qkv' in name) or (isinstance(module, nn.Linear) and module.out_features == module.in_features):
            if module.weight.grad is not None:
                metric_array.append(torch.abs(
                        torch.norm(module.weight.grad, 'nuc') * torch.norm(module.weight, 'nuc')))
            else:
                metric_array.append(torch.zeros_like(module.weight))

        if isinstance(module, nn.Linear) and 'qkv' not in name and module.out_features != module.in_features and 'head' not in name:
            if module.weight.grad is not None:
                metric_array.append(torch.abs(module.weight.grad * module.weight))
            else:
                metric_array.append(torch.zeros_like(module.weight))

        elif isinstance(module, torch.nn.Linear) and 'head' in name:
            if module.weight.grad is not None:
                metric_array.append(torch.abs(module.weight.grad * module.weight))
            else:
                metric_array.append(torch.zeros_like(module.weight))

    return metric_array







def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(
                torch.stack([torch.Tensor(sh).fill_(v) for v in e],
                            dim=0).to(device))
        return ret_grads

    if type(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
        return outer
    else:
        return broadcast_val(elements, shapes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_2(model):
    num = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                num += layer.weight.grad.data.numpy().shape[0]
            else:
                num += layer.weight.shape[0]
    return num


def get_flattened_metric(net, metric, verbose=False):
    grad_list = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grad_list.append(metric(layer).flatten())
            if verbose:
                print(layer.__class__.__name__, metric(layer).flatten()[:10])
    return np.concatenate(grad_list)
