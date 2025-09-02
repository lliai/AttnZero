# self-attention
# kernel linear attention:
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py


import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn, einsum

from autoatten.operators.graph_ops import  get_graph_candidates



class AutoAttention_deit(nn.Module):

    def __init__(self, atten_cfg, num_patches,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0., ):
        super(AutoAttention_deit, self).__init__()


        self.dim = dim

        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size

        self.num_heads = num_heads
        self.atten_cfg = atten_cfg
        self.transform = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)

    def forward_diag(self, input):
        res=get_graph_candidates(
            self.atten_cfg['graph_type'], input,
            self.atten_cfg
        )
        return res


    def forward(self, x):
        N, L, C = x.shape
        h = int(L ** 0.5)
        w = int(L ** 0.5)
        num_heads = self.num_heads
        head_dim = C // num_heads

        x = self.transform(x).view(N, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]

        diag_inputs =[q,k,v]


        out = self.forward_diag(diag_inputs)

        out = out.transpose(1, 2).contiguous().view(N, L, C)

        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(N, h, w, C).permute(0, 3, 1, 2)
        out[:, 1:, :] = out[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(N, L - 1, C)

        out = self.projection(out)
        out = self.projection_dropout(out)

        return out



class AutoAttention_pvt(nn.Module):

    def __init__(self, atten_cfg, num_patches,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 sr_ratio=1):
        super(AutoAttention_pvt, self).__init__()


        self.dim = dim
        # print('dim//num_heads:', dim//num_heads)

        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size

        self.num_heads = num_heads
        self.atten_cfg = atten_cfg
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=dim, out_channels= dim, kernel_size=(3, 3),
                             padding=1, groups= dim)



    def forward_diag(self, input):
        res=get_graph_candidates(
            self.atten_cfg['graph_type'], input,
            self.atten_cfg
        )
        return res


    def forward(self, x, H, W):
        N, L, C = x.shape
        num_heads = self.num_heads
        head_dim = C // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(N, C, H, W)
            x_ = self.sr(x_).reshape(N, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(N, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(N, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        q = q.reshape(N, L, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, L // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, L // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)

        out = self.forward_diag([q,k,v])

        out = out.transpose(1, 2).contiguous().view(N, L, C)

        v_ = v.transpose(1, 2).reshape(N, H // self.sr_ratio, W // self.sr_ratio, C).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v_ = nn.functional.interpolate(v_, size=(H, W), mode='bilinear')
        out = out + self.dwc(v_).permute(0, 2, 3, 1).reshape(N, L, C)

        out = self.projection(out)
        out = self.projection_dropout(out)

        return out






class AutoAttention_pvt_v2(nn.Module):

    def __init__(self, atten_cfg, num_patches,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 sr_ratio=1,
                 linear=False):
        super(AutoAttention_pvt_v2, self).__init__()


        self.dim = dim

        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.linear = linear

        self.num_heads = num_heads
        self.atten_cfg = atten_cfg
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.dwc = nn.Conv2d(in_channels=dim, out_channels= dim, kernel_size=(3, 3),
                             padding=1, groups= dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def forward_diag(self, input):
        res=get_graph_candidates(
            self.atten_cfg['graph_type'], input,
            self.atten_cfg
        )
        return res


    def forward(self, x, H, W):
        N, L, C = x.shape
        num_heads = self.num_heads
        head_dim = C // num_heads
        q = self.q(x)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(N, C, H, W)
                x_ = self.sr(x_).reshape(N, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(N, -1, 2, C).permute(2, 0, 1, 3)
            else:
                kv = self.kv(x).reshape(N, -1, 2, C).permute(2, 0, 1, 3)
        else:
            x_ = x.permute(0, 2, 1).reshape(N, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(N, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(N, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        q = q.reshape(N, L, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, L // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, L // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)

        out = self.forward_diag([q,k,v])

        out = out.transpose(1, 2).contiguous().view(N, L, C)

        v_ = v.transpose(1, 2).reshape(N, H // self.sr_ratio, W // self.sr_ratio, C).permute(0, 3, 1, 2)
        if self.sr_ratio > 1 or self.linear:
            v_ = nn.functional.interpolate(v_, size=(H, W), mode='bilinear')
        out = out + self.dwc(v_).permute(0, 2, 3, 1).reshape(N, L, C)

        out = self.projection(out)
        out = self.projection_dropout(out)

        return out






class AutoAttention_swin(nn.Module):

    def __init__(self, atten_cfg, window_size,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 ):

        super(AutoAttention_swin, self).__init__()


        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.atten_cfg = atten_cfg

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

        self.dwc = nn.Conv2d(in_channels=dim, out_channels= dim, kernel_size=(3, 3),
                             padding=1, groups= dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def forward_diag(self, input):
        res=get_graph_candidates(
            self.atten_cfg['graph_type'], input,
            self.atten_cfg
        )
        return res


    def forward(self, x, mask=None):
        N, L, C = x.shape
        num_heads = self.num_heads
        head_dim = C // num_heads
        h = int(L ** 0.5)
        w = int(L ** 0.5)
        qkv = self.qkv(x).reshape(N, L, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.reshape(N, L, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, L, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, L, num_heads, head_dim).permute(0, 2, 1, 3)


        out = self.forward_diag([q,k,v])

        out = out.transpose(1, 2).contiguous().view(N, L, C)

        v_ = v.transpose(1, 2).reshape(N, h , w , C).permute(0, 3, 1, 2)
        out = out + self.dwc(v_).permute(0, 2, 3, 1).reshape(N, L, C)

        out = self.projection(out)
        out = self.projection_dropout(out)

        return out





