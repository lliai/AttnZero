import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
import numpy as np
from autoatten.operators import get_graph_candidates
from autoatten.operators import get_graph_candidates, unary_operation, binary_operation

def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))




class AF_Vision_Transformer(nn.Module):

    def __init__(self, atten_cfg = None, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=[3]*12, mlp_ratio=[4.]*12, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., pre_norm=True, scale=False, gp=True, relative_position=True, abs_pos = True, max_relative_position=14):
        super(AF_Vision_Transformer, self).__init__()
        # the configs of super arch
        self.embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.mlp_ratio = mlp_ratio
        self.layer_num = depth
        self.num_heads = num_heads
        self.dropout = drop_rate
        self.attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm=pre_norm
        self.scale=scale
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim)
        self.gp = gp


        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(atten_cfg= atten_cfg, dim=embed_dim, num_heads=num_heads[i], mlp_ratio=mlp_ratio[i],
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop_rate,
                                                       attn_drop=attn_drop_rate, drop_path=dpr[i],
                                                       pre_norm=pre_norm, scale=self.scale,
                                                        relative_position=relative_position,
                                                       max_relative_position=max_relative_position))

        # parameters for vision transformer
        num_patches = self.patch_embed.num_patches

        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = nn.LayerNorm(embed_dim)


        # classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()




    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed

        x = F.dropout(x, p=self.dropout, training=self.training)

        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
        # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x)

        if self.gp:
            return torch.mean(x[:, 1:] , dim=1)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x





def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)



class RelativePosition2D(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)


    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        # compute the row and column distance
        distance_mat_v = (range_vec_k[None, :] // int(length_q ** 0.5 )  - range_vec_q[:, None] // int(length_q ** 0.5 ))
        distance_mat_h = (range_vec_k[None, :] % int(length_q ** 0.5 ) - range_vec_q[:, None] % int(length_q ** 0.5 ))
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, -self.max_relative_position, self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, -self.max_relative_position, self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1,0,1,0), "constant", 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1,0,1,0), "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings









class AutoAttention(nn.Module):
    def __init__(self, atten_cfg, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., normalization = False, relative_position = True,
                 num_patches = None, max_relative_position=14, scale=False):
        super().__init__()
        self.atten_cfg = atten_cfg
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.embed_dim = embed_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        # if self.relative_position:
        #     # self.rel_pos_embed_k = RelativePosition2D(embed_dim //num_heads, max_relative_position)
        #     self.rel_pos_embed_v = RelativePosition2D(embed_dim //num_heads, max_relative_position)
        self.max_relative_position = max_relative_position


        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwc = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3),
                             padding=1, groups=embed_dim)

    def forward_diag(self, x):
        res = get_graph_candidates(
            self.atten_cfg['graph_type'], x,
            self.atten_cfg
        )
        return res


    def forward(self, x):
        B, N, C = x.shape
        h = int(N ** 0.5)
        w = int(N ** 0.5)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

       #  forward diag ------------------------------------------------------------------------
       #  attn = (q @ k.transpose(-2, -1)) * self.sample_scale
       #  if self.relative_position:
       #      r_p_k = self.rel_pos_embed_k(N, N)
       #      attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)) \
       #          .transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.sample_scale
       #  attn = attn.softmax(dim=-1)
       #  attn = self.attn_drop(attn)
       #  x = (attn @ v).transpose(1,2).reshape(B, N, -1)
        #  forward diag ------------------------------------------------------------------------

        x = self.forward_diag([q,k,v])

        # if self.relative_position:
        #     r_p_v = self.rel_pos_embed_v(N, N)
        #     attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.num_heads, -1)
        #     # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
        #     # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
        #     # same size as x (B, num_heads, N, hidden_dim)
        #     x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B, self.num_heads, N, -1).transpose(2,1).reshape(B, N, -1)

        x = x.transpose(1, 2).reshape(B, N, C)
        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(B, h, w, C).permute(0, 3, 1, 2)
        x[:, 1:, :] = x[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(B, N - 1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, atten_cfg, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pre_norm=True, scale=False,
                 relative_position=True,  max_relative_position=14):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.embed_dim = dim
        self.mlp_ratio = mlp_ratio
        self.ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.num_heads = num_heads
        self.normalize_before = pre_norm
        self.dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position


        self.is_identity_layer = None
        self.attn = AutoAttention(atten_cfg = atten_cfg,
            embed_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position,
            max_relative_position=max_relative_position
        )

        self.attn_layer_norm = norm_layer(dim)
        self.ffn_layer_norm = norm_layer(dim)
        # self.dropout = dropout
        self.activation_fn = gelu


        self.fc1 = nn.Linear(dim, self.ffn_embed_dim_this_layer)
        self.fc2 = nn.Linear(self.ffn_embed_dim_this_layer, dim)




    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)
        # compute the ffn
        # start_time = time.time()
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


def calc_dropout(dropout, embed_dim):
    return dropout * 1.0 * embed_dim