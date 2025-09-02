from __future__ import print_function
import math
import random
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

# x=torch.rand((10,3,224,224))
# x =  rearrange(x, 'b c (h hp) (w wp) -> b c (h w) hp wp',hp=16,wp=16)#
'''
knowledge:
batch: feature =  rearrange(feature, 'b c h w -> b (c h w) ')
channel: feature =  rearrange(feature, 'b c h w -> b c (h w)')
spatial: feature =  rearrange(feature, b c (h hp) (w wp) -> b (c h w) hp wp',hp=4,wp=4)
intra-class: logit =  rearrange(logit, n c -> n c ')
inter-class: logit =  rearrange(logit, n c -> c n ')
distance:
l2,
gmml2,
KL,
cosine
loss:
batch_gmml2, batch_kl;
channel_gmml2, channel_kl;
Spatial_kl;
Inter_class_cosine;
Intra_class_cosine;
knowledge_list = [batch, channel, spatial, intra-class, inter-class]
distance_list =  [l2, gmml2, KL, cosine]
loss_weight_list = [0.01, 0.1, 1, 10, 100, 1000]

'''


class Batch_gmml2(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""

    def __init__(self):
        super(Batch_gmml2, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        return (G_diff * G_diff).view(-1, 1).sum() / (bsz * bsz)


class KnowledgeEncoder(nn.Module):
    """Encode different types of knowledge."""

    def __init__(self) -> None:
        super().__init__()

        self.knowledge_form = {
            'none':
            lambda x: x,
            'batch':
            lambda x: rearrange(x, 'b c h w -> b (c h w)'),
            'channel':
            lambda x: rearrange(x, 'b c h w -> b c (h w)'),
            'spatial':
            lambda x: rearrange(
                x, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=4, wp=4)
        }

        self.distance_list = [
            nn.L1Loss(),
            nn.MSELoss(),
            nn.KLDivLoss(reduction='batchmean')
        ]

    def forward(self, preds_S, preds_T):
        ...


class Batch_kl(nn.Module):

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(Batch_kl, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.
        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).
        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        softmax_pred_T = F.softmax(
            preds_T.view(-1, C * W * H) / self.tau, dim=0)

        logsoftmax = torch.nn.LogSoftmax(dim=0)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, C * W * H) / self.tau) -
                         softmax_pred_T * logsoftmax(
                             preds_S.view(-1, C * W * H) / self.tau)) * (
                                 self.tau**2)

        return self.loss_weight * loss / N


class Channel_gmml2(nn.Module):
    """Inter-Channel Correlation"""

    def __init__(self, ):
        super(Channel_gmml2, self).__init__()

    def forward(self, f_s, f_t):

        bsz, ch = f_s.shape[0], f_s.shape[1]

        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)

        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        G_diff = emd_s - emd_t
        loss = (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)
        return loss


class Channel_kl(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(Channel_kl, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.
        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).
        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)
        loss = self.loss_weight * loss / (C * N)
        return loss


class Spatial_kl(nn.Module):

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(Spatial_kl, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.
        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).
        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        preds_S = rearrange(
            preds_S, 'b c (h hp) (w wp) -> b c (h w) hp wp', hp=4, wp=4)
        preds_T = rearrange(
            preds_T, 'b c (h hp) (w wp) -> b c (h w) hp wp', hp=4, wp=4)
        preds_S = rearrange(preds_S, 'b c n hp wp -> b (c n) hp wp')
        preds_T = rearrange(preds_T, 'b c n hp wp -> b (c n) hp wp')
        N, C, H, W = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)
        return self.loss_weight * loss / (C * N)


class Inter_class_cosine(nn.Module):

    def __init__(self, beta=1.0, gamma=1.0):
        super(Inter_class_cosine, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = z_s.softmax(dim=1)
        y_t = z_t.softmax(dim=1)
        return inter_class_relation(y_s, y_t)


class Intra_class_cosine(nn.Module):

    def __init__(self, beta=1.0, gamma=1.0):
        super(Inter_class_cosine, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = z_s.softmax(dim=1)
        y_t = z_t.softmax(dim=1)
        return intra_class_relation(y_s, y_t)


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):

    def __init__(self, beta=1.0, gamma=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = z_s.softmax(dim=1)
        y_t = z_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        return self.beta * inter_loss + self.gamma * intra_loss


class AutoKD_v0(nn.Module):
    """Demo of make coefficient learnable"""

    def __init__(self, loss_weight_list=None) -> None:
        if loss_weight_list is None:
            loss_weight_list = [0.01, 0.1, 1, 10, 100, 1000]

        super().__init__()
        self.loss_weight_list = loss_weight_list
        self.auto_scale = nn.Parameter(
            torch.randn(1) * 1e-4, requires_grad=True)
        # here we don't make hyperparameter of loss learnable.
        # only the coefficient before loss is learnable.
        self.demo_loss = DIST()

    def forward(self, pred_s, pred_t):
        auto_scale = torch.sigmoid(self.auto_scale) * 1
        return auto_scale * self.demo_loss(pred_s, pred_t)


class AutoKD_v1(nn.Module):
    """Auto Learning KD coefficient.

    Args:
        loss_weight (List, optional): _description_. Defaults to None.
    """

    def __init__(self, loss_weight: List = None) -> None:
        super().__init__()
        if loss_weight is None:
            # prior
            loss_weight = [0.125, 0.25, 0.5, 0.125]
        self.loss_weight = loss_weight

        # self.feat_level_scale = nn.Parameter(torch.randn(4) * 1e-4,
        #                                      requires_grad=True)
        # results: 73.14
        self.register_parameter(
            'feat_level_scale',
            nn.Parameter(torch.randn(4) * 1e-4, requires_grad=True))
        # TODO
        # self.logit_level_scale = nn.Parameter(torch.randn(1)*1e-4, requires_grad=True)

        self.kd_loss_pool = [
            Spatial_kl(),
            Channel_kl(),
            Channel_gmml2(),
            Batch_kl()
        ]
        assert len(self.loss_weight) == len(self.kd_loss_pool)

    def forward(self, pred_s, pred_t):
        """Forward of autokd.

        Args:
            pred_s (Tensor): last feature map of student net
            pred_t (Tensor): last feature map of teacher net
        """
        # self.feat_level_scale = F.softmax(self.feat_level_scale, dim=-1)
        sum_loss = 0
        for w, l, s in zip(self.loss_weight, self.kd_loss_pool,
                           self.feat_level_scale):
            sum_loss += torch.sigmoid(s) * w * l(pred_s, pred_t)
        return sum_loss


class AutoKD_v2(nn.Module):
    """Auto Learning KD coefficient.

    modify parameter initialize way.

    Args:
        loss_weight (List, optional): _description_. Defaults to None.
    """

    def __init__(self, loss_weight: List = None, num_loss: int = 4) -> None:
        super().__init__()
        if loss_weight is None:
            # prior
            loss_weight = [0.125, 0.25, 0.5, 0.125]
        self.loss_weight = loss_weight

        # self.feat_level_scale = nn.Parameter(torch.randn(4) * 1e-4,
        #                                      requires_grad=True)
        self.register_parameter(
            'feat_level_scale',
            nn.Parameter(
                nn.init.uniform_(torch.empty(num_loss), a=0.2, b=1.0),
                requires_grad=True))
        # TODO
        # self.logit_level_scale = nn.Parameter(torch.randn(1)*1e-4, requires_grad=True)

        self.kd_loss_pool = [
            Spatial_kl(),
            Channel_kl(),
            Channel_gmml2(),
            Batch_kl()
        ]
        assert len(self.loss_weight) == len(self.kd_loss_pool)

    def forward(self, pred_s, pred_t):
        """Forward of autokd.

        Args:
            pred_s (Tensor): last feature map of student net
            pred_t (Tensor): last feature map of teacher net
        """
        # self.feat_level_scale = F.softmax(self.feat_level_scale, dim=-1)
        sum_loss = 0
        for w, l, s in zip(self.loss_weight, self.kd_loss_pool,
                           self.feat_level_scale):
            sum_loss += torch.sigmoid(s) * w * l(pred_s, pred_t)
        return sum_loss


class AutoKD_v3(nn.Module):
    """Auto Learning KD coefficient.
    Each param corresponds to a list of coefficient.

    modify parameter initialize way.

    Args:
        loss_weight (List, optional): _description_. Defaults to None.
    """

    def __init__(self, loss_weight: List = None, num_loss: int = 4) -> None:
        super().__init__()
        if loss_weight is None:
            # prior
            loss_weight = [0.01, 0.1, 1, 10, 100, 1000]
        self.loss_weight = loss_weight

        self.register_parameter(
            'feat_level_scale',
            nn.Parameter(
                nn.init.uniform_(
                    torch.empty(num_loss, len(loss_weight)), a=0.2, b=1.0),
                requires_grad=True))

        self.kd_loss_pool = [
            Spatial_kl(),
            Channel_kl(),
            Channel_gmml2(),
            Batch_kl()
        ]

    def forward(self, pred_s, pred_t):
        """Forward of autokd.

        Args:
            pred_s (Tensor): last feature map of student net
            pred_t (Tensor): last feature map of teacher net
        """
        # self.feat_level_scale = F.softmax(self.feat_level_scale, dim=-1)
        sum_loss = 0
        for l, params in zip(self.kd_loss_pool, self.feat_level_scale):
            for s, param in zip(self.loss_weight, params):
                sum_loss += s * torch.sigmoid(param) * l(pred_s, pred_t)
        return sum_loss


class AutoKD_v4(nn.Module):
    """Auto Learning KD coefficient.
    Each param corresponds to a list of coefficient.

    Note:
        Using Softmax or Gumbel Softmax.
        modify parameter initialize way.
        We prefer using softmax during the searching phase.
        delete sigmoid


    Args:
        loss_weight (List, optional): _description_. Defaults to None.
    """

    def __init__(self, loss_weight: List = None, num_loss: int = 4) -> None:
        super().__init__()
        if loss_weight is None:
            # prior
            loss_weight = [0.01, 0.1, 1, 10]
        self.loss_weight = loss_weight

        self.register_parameter(
            'feat_level_scale',
            nn.Parameter(
                nn.init.uniform_(
                    torch.empty(num_loss, len(loss_weight)), a=0.2, b=1.0),
                requires_grad=True))

        self.kd_loss_pool = {
            'spatial_kl': Spatial_kl(),
            'channel_kl': Channel_kl(),
            'channel_gmml2': Channel_gmml2(),
            'batch_kl': Batch_kl()
        }

    def forward(self, pred_s, pred_t):
        """Forward of autokd."""
        sum_loss = 0
        for lk, params in zip(self.kd_loss_pool, self.feat_level_scale):
            soft_params = F.softmax(params, dim=-1)
            for s, param in zip(self.loss_weight, soft_params):
                sum_loss += s * torch.sigmoid(param) * self.kd_loss_pool[lk](
                    pred_s, pred_t)
        return sum_loss


class AutoKD_v5(nn.Module):
    """Auto Learning KD coefficient.
    Each param corresponds to a list of coefficient.

    Note:
        Using Softmax or Gumbel Softmax.
        modify parameter initialize way.
        We prefer using softmax during the searching phase.
        Add parameter for loss level.

    Args:
        loss_weight (List, optional): _description_. Defaults to None.
    """

    def __init__(self, loss_weight: List = None, num_loss: int = 4) -> None:
        super().__init__()
        if loss_weight is None:
            # prior
            loss_weight = [0.01, 0.1, 1, 10]
        self.loss_weight = loss_weight

        self.register_parameter(
            'feat_level_scale',
            nn.Parameter(
                nn.init.uniform_(
                    torch.empty(num_loss, len(loss_weight)), a=0.2, b=1.0),
                requires_grad=True))
        self.register_parameter(
            'loss_level_scale',
            nn.Parameter(
                nn.init.uniform_(torch.empty(num_loss), a=0.2, b=1.0),
                requires_grad=True))

        self.kd_loss_pool = {
            'spatial_kl': Spatial_kl(),
            'channel_kl': Channel_kl(),
            'channel_gmml2': Channel_gmml2(),
            'batch_kl': Batch_kl()
        }

    def forward(self, pred_s, pred_t):
        """Forward of autokd.

        Args:
            pred_s (Tensor): last feature map of student net
            pred_t (Tensor): last feature map of teacher net
        """
        sum_loss = 0
        loss_level_scale = F.softmax(self.loss_level_scale, dim=-1)
        for lk, params, lls in zip(self.kd_loss_pool, self.feat_level_scale,
                                   loss_level_scale):
            soft_params = F.softmax(params, dim=-1)
            for s, param in zip(self.loss_weight, soft_params):
                sum_loss += lls * s * torch.sigmoid(
                    param) * self.kd_loss_pool[lk](pred_s, pred_t)
        return sum_loss


AutoKD = AutoKD_v5
