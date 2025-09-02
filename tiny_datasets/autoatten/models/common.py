import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch
from autoatten.core.config import cfg


def layernorm(w_in):
    return nn.LayerNorm(w_in, eps=cfg.TRANSFORMER.LN_EPS)




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
