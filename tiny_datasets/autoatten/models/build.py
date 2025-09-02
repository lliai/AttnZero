from fvcore.common.registry import Registry

from autoatten.core.config import cfg
from .distill import DistillationWrapper
from .autoatten_pvt import AutoAtten_pvt_tiny
from .autoatten_pvt_v2 import AutoAtten_pvt_v2_b0
from .autoatten_swin import AutoAttenSwinTransformer
from .autoformer import AutoFormer

MODEL = Registry('MODEL')


def build_model(**kwargs):
    model_type = cfg.MODEL.TYPE
    genotype_keys = ('graph_type', 'unary_op', 'binary_op', 'bias_index')
    atten_cfg = dict.fromkeys(genotype_keys)
    atten_cfg['graph_type'] = cfg.AT.graph_type
    atten_cfg['unary_op'] = cfg.AT.unary_op
    atten_cfg['binary_op'] = cfg.AT.binary_op
    if model_type in ['AutoAtten_pvt_tiny', 'AutoAtten_pvt_v2_b0']:
        model = eval(model_type + '(drop_path_rate=cfg.MODEL.DROP_PATH_RATE,'
                                  'num_classes=cfg.MODEL.NUM_CLASSES,'
                                  'attn_type=cfg.MODEL.STAGE_ATTN_TYPE,'
                                  'atten_cfg=atten_cfg,'
                                  'la_sr_ratios=str(cfg.PVT.LA_SR_RATIOS))')
    elif model_type in ['AutoAtten_swin_tiny']:

        model = AutoAttenSwinTransformer(
            img_size=cfg.MODEL.IMG_SIZE,
            patch_size=cfg.SWIN.PATCH_SIZE,
            in_chans=cfg.SWIN.IN_CHANS,
            num_classes=cfg.MODEL.NUM_CLASSES,
            embed_dim=cfg.SWIN.EMBED_DIM,
            depths=cfg.SWIN.DEPTHS,
            num_heads=cfg.SWIN.NUM_HEADS,
            window_size=cfg.SWIN.WINDOW_SIZE,
            mlp_ratio=cfg.SWIN.MLP_RATIO,
            qkv_bias=cfg.SWIN.QKV_BIAS,
            qk_scale=cfg.SWIN.QK_SCALE,
            drop_rate=cfg.MODEL.DROP_RATE,
            drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
            ape=cfg.SWIN.APE,
            patch_norm=cfg.SWIN.PATCH_NORM,
            use_checkpoint=cfg.TRAIN.USE_CHECKPOINT,
            attn_type=cfg.MODEL.STAGE_ATTN_TYPE,
            atten_cfg=atten_cfg
        )
    elif model_type in ['AutoFormer', 'PIT']:

        model = AutoFormer(atten_cfg, num_classes=cfg.MODEL.NUM_CLASSES)

    return model


