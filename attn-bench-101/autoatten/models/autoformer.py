import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .build import MODEL
from autoatten.core.config import cfg
from timm.models.layers import DropPath,  trunc_normal_
import numpy as np
from autoatten.operators import get_graph_candidates



def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def layernorm(w_in):
    return nn.LayerNorm(w_in, eps=1e-6)




class MLP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 drop_rate=0.,
                 hidden_ratio=1.):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = int(in_channels * hidden_ratio)
        self.fc1 = nn.Linear(in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, out_channels)
        self.drop = nn.Dropout(drop_rate)
        self.activation_fn = torch.nn.GELU()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class PatchEmbedding(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_channels=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AutoAttention(nn.Module):

    def __init__(self, atten_cfg,
                 in_channels,
                 out_channels,
                 num_heads,
                 qkv_bias=False,
                 attn_drop_rate=0.,
                 proj_drop_rate=0., ):
        super(AutoAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.atten_cfg = atten_cfg
        self.transform = nn.Linear(in_channels, out_channels * 3, bias=qkv_bias)

        self.projection = nn.Linear(out_channels, out_channels)
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.projection_dropout = nn.Dropout(proj_drop_rate)

        self.dwc = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                             padding=1, groups=out_channels)


    def forward_diag(self, x):
        res=get_graph_candidates(
            self.atten_cfg['graph_type'], x,
            self.atten_cfg
        )
        return res


    def forward(self, x):
        N, L, C = x.shape
        h = int(L ** 0.5)
        w = int(L ** 0.5)

        x = self.transform(x).view(N, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
         # -------------------------- forward_diag ---------------------------------------
        # query, key, value = x[0], x[1], x[2]
        # # print('query:', query.size())    torch.Size([128, 3, 197, 64])
        #
        # qk = query @ key.transpose(-1, -2) * self.norm_factor
        # # print('qk:', qk.size())   torch.Size([128, 3, 197, 197])
        # qk = F.softmax(qk, dim=-1)
        # qk = self.attention_dropout(qk)
        #
        # out = qk @ value

        # -------------------------- forward_diag ---------------------------------------

        out = self.forward_diag([q, k, v])

        # print('qk @ value:', out.size())     torch.Size([128, 3, 197, 64])
        out = out.transpose(1, 2).contiguous().view(N, L, self.out_channels)
        # print('before projection:', out.size())    torch.Size([128, 197, 192])

        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(N, h, w, C).permute(0, 3, 1, 2)
        out[:, 1:, :] = out[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(N, L - 1, C)


        out = self.projection(out)
        out = self.projection_dropout(out)

        return out




class TransformerLayer(nn.Module):

    def __init__(self,
                 atten_cfg,
                 in_channels,
                 num_heads,
                 qkv_bias=False,
                 out_channels=None,
                 mlp_ratio=1.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qk_scale=None):
        super(TransformerLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = layernorm(in_channels)

        self.attn = AutoAttention(atten_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = layernorm(out_channels)
        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            drop_rate=drop_rate,
            hidden_ratio=mlp_ratio)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x








# @MODEL.register()
class AutoFormer(nn.Module):

    def __init__(self, img_size =224, patch_size=16, in_channels=3, attn_drop_rate =0.0, drop_path_rate = 0.0, drop_rate =0.0,
                 atten_config = None, num_classes = None):
        super(AutoFormer, self).__init__()
        # the configs of super arch
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_heads = cfg.AUTOFORMER.NUM_HEADS
        self.mlp_ratio = cfg.AUTOFORMER.MLP_RATIO
        self.hidden_dim = cfg.AUTOFORMER.HIDDEN_DIM
        self.depth = cfg.AUTOFORMER.DEPTH

        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        if atten_config:
            self.atten_config = atten_config

        else:
            self.atten_config = {}
            self.atten_config['graph_type'] = cfg.AT.GRAPH_TYPE
            self.atten_config['unary_op'] = cfg.AT.UNARY_OP
            self.atten_config['binary_op'] = cfg.AT.BINARY_OP

        if num_classes:
            self.num_classes = num_classes
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES

        self.feature_dims = [self.hidden_dim] * self.depth

        self.patch_embed = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels, out_channels=self.hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_tokens = 1



        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule

        for i in range(self.depth):
            self.blocks.append(TransformerLayer(self.atten_config, in_channels=self.hidden_dim, num_heads=self.num_heads[i], qkv_bias=True,
                                                       mlp_ratio=self.mlp_ratio[i], drop_rate=self.drop_rate,
                                                       attn_drop_rate=self.attn_drop_rate, drop_path_rate=dpr[i],
                                                       ))


        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.hidden_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = layernorm(self.hidden_dim)


        # classifier head
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

        self.apply(self._init_weights)


        self.distill_logits = None

        self.distill_token = None
        self.distill_head = None





    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.patch_embed(x)
        if self.num_tokens == 1:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)
        else:
            x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), self.distill_token.repeat(x.size(0), 1, 1), x], dim=1)

        x = x + self.pos_embed
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return torch.mean(x[:, 1:] , dim=1)



    def forward(self, x):
        x = self.forward_features(x)
        logits = self.head(x)
        if self.num_tokens == 1:
            return logits

        self.distill_logits = None
        self.distill_logits = self.distill_head(x)

        if self.training:
            return logits
        else:
            return (logits + self.distill_logits) / 2




