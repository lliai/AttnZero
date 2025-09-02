

from autoatten.core.config import cfg
from .autoformer import AutoFormer
from .autoatten_deit import AutoAtten_deit_tiny
from .autoatten_pvt import AutoAtten_pvt_tiny
from .autoatten_pvt_v2 import AutoAtten_pvt_v2_b0
from .autoatten_swin import AutoAttenSwinTransformer




def build_model(config):
    model_type = config.MODEL.TYPE
    genotype_keys = ('graph_type', 'unary_op', 'binary_op')
    atten_cfg = dict.fromkeys(genotype_keys)
    atten_cfg['graph_type'] = config.AT.GRAPH_TYPE
    atten_cfg['unary_op'] = config.AT.UNARY_OP
    atten_cfg['binary_op'] = config.AT.BINARY_OP

    # if model_type in ['flatten_pvt_tiny',   'flatten_pvt_v2_b0',
    #                   'Primal_pvt_tiny', 'Primal_pvt_v2_b0',
    #                   ]:
    #
    #     model = eval(model_type + '(drop_path_rate=config.MODEL.DROP_PATH_RATE,'
    #                               'num_classes=config.MODEL.NUM_CLASSES,'
    #                               'attn_type=config.MODEL.STAGE_ATTN_TYPE,'
    #                               'la_sr_ratios=str(config.PVT.LA_SR_RATIOS))')



    # elif model_type in ['flatten_deit_tiny', 'Primal_deit_tiny']:
    #     model = eval(model_type + '(num_classes=config.MODEL.NUM_CLASSES,'
    #                               'drop_path_rate=config.MODEL.DROP_PATH_RATE)')





    if model_type in ['AutoAtten_pvt_tiny', 'AutoAtten_pvt_v2_b0'
                        ]:
        model = eval(model_type + '(drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'attn_type=config.MODEL.STAGE_ATTN_TYPE,'
                                  'atten_cfg=atten_cfg,'
                                  'la_sr_ratios=str(config.PVT.LA_SR_RATIOS))')

    elif model_type in ['AutoAtten_deit_tiny']:
        model = eval(model_type + '(num_classes=config.MODEL.NUM_CLASSES,'
                                  'atten_cfg=atten_cfg,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE)')

    elif model_type in ['AutoAtten_swin_tiny']:

        model = AutoAttenSwinTransformer(
            img_size=config.MODEL.IMG_SIZE,
            patch_size=config.SWIN.PATCH_SIZE,
            in_chans=config.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.SWIN.EMBED_DIM,
            depths=config.SWIN.DEPTHS,
            num_heads=config.SWIN.NUM_HEADS,
            window_size=config.SWIN.WINDOW_SIZE,
            mlp_ratio=config.SWIN.MLP_RATIO,
            qkv_bias=config.SWIN.QKV_BIAS,
            qk_scale=config.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.SWIN.APE,
            patch_norm=config.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            attn_type=config.MODEL.STAGE_ATTN_TYPE,
            atten_cfg=atten_cfg
        )

    elif model_type in ['AutoFormer', 'PIT']:

        model = AutoFormer(drop_path_rate=cfg.MODEL.DROP_PATH_RATE, atten_config=atten_cfg, num_classes=cfg.MODEL.NUM_CLASSES)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    #
    return model





