# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from .autoatten_autoformer import AF_Vision_Transformer
from .autoatten_swin import AutoAttenSwinTransformer
from .autoatten_swin_bias import AutoAttenSwinTransformer_Bias
from .autoatten_deit import AutoAtten_deit_tiny, AutoAtten_deit_small, AutoAtten_deit_base
from .autoatten_deit_bias import AutoAtten_deit_tiny_bias
from .autoatten_pvt import AutoAtten_pvt_tiny, AutoAtten_pvt_small, AutoAtten_pvt_medium, AutoAtten_pvt_large
from .AT_105_pvt_bias import AutoAtten_Bias_pvt_tiny
from .autoatten_cswin import AutoAtten_CSWin_64_24181_tiny_224, AutoAtten_CSWin_96_36292_base_224, \
                             AutoAtten_CSWin_64_36292_small_224, AutoAtten_CSWin_96_36292_base_384


def build_model(config):
    model_type = config.MODEL.TYPE
    genotype_keys = ('graph_type', 'unary_op', 'binary_op')
    atten_cfg = dict.fromkeys(genotype_keys)
    atten_cfg['graph_type'] = config.MODEL.AT.GRAPH_TYPE
    atten_cfg['unary_op'] = config.MODEL.AT.UNARY_OP
    atten_cfg['binary_op'] = config.MODEL.AT.BINARY_OP
    if model_type == 'autoatten_swin':
        model = AutoAttenSwinTransformer(img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                     in_chans=config.MODEL.SWIN.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                     depths=config.MODEL.SWIN.DEPTHS,
                                     num_heads=config.MODEL.SWIN.NUM_HEADS,
                                     window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                     qk_scale=config.MODEL.SWIN.QK_SCALE,
                                     drop_rate=config.MODEL.DROP_RATE,
                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                     ape=config.MODEL.SWIN.APE,
                                     patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                     attn_type=config.MODEL.STAGE_ATTN_TYPE,
                                     atten_cfg=atten_cfg)
    elif model_type == 'autoatten_swin_bias':
        model = AutoAttenSwinTransformer_Bias(img_size=config.DATA.IMG_SIZE,
                                         patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                         in_chans=config.MODEL.SWIN.IN_CHANS,
                                         num_classes=config.MODEL.NUM_CLASSES,
                                         embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                         depths=config.MODEL.SWIN.DEPTHS,
                                         num_heads=config.MODEL.SWIN.NUM_HEADS,
                                         window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                         qk_scale=config.MODEL.SWIN.QK_SCALE,
                                         drop_rate=config.MODEL.DROP_RATE,
                                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                         ape=config.MODEL.SWIN.APE,
                                         patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                         use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                         attn_type=config.MODEL.STAGE_ATTN_TYPE,
                                         atten_cfg=atten_cfg)
    #
    elif model_type in ['AutoAtten_deit_tiny', 'AutoAtten_deit_small', 'AutoAtten_deit_base', 'AutoAtten_deit_tiny_bias']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'atten_cfg=atten_cfg)')
    #
    elif model_type in ['AutoAtten_pvt_tiny', 'AutoAtten_pvt_small', 'AutoAtten_pvt_medium', 'AutoAtten_pvt_large', 'AutoAtten_Bias_pvt_tiny']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'attn_type=config.MODEL.STAGE_ATTN_TYPE,'
                                  'agent_sr_ratios=str(config.MODEL.PVT_LA_SR_RATIOS),'
                                  'atten_cfg=atten_cfg)')

    elif model_type in ['AutoAtten_CSWin_64_24181_tiny_224',
                        'AutoAtten_CSWin_96_36292_base_224', 'AutoAtten_CSWin_64_36292_small_224',
                        'AutoAtten_CSWin_96_36292_base_384'
                        ]:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'attn_type=config.MODEL.STAGE_ATTN_TYPE,'
                                  'atten_cfg=atten_cfg,'
                                  'la_split_size=config.MODEL.CSWIN_LA_SPLIT_SIZE)')
    elif model_type in ['AutoFormer']:
        model = AF_Vision_Transformer(atten_cfg = atten_cfg,
                                      embed_dim=config.MODEL.AUTOFORMER.HIDDEN_DIM,
                                      depth=config.MODEL.AUTOFORMER.DEPTH,
                                      num_heads=config.MODEL.AUTOFORMER.NUM_HEADS,
                                      mlp_ratio=config.MODEL.AUTOFORMER.MLP_RATIO)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
