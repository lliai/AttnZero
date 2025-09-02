_base_ = [
    '../../_base_/models/fpn_r50.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='../Detection/pretrained/AutoAtten_pvt_small_max_acc.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth',
    backbone=dict(
        type='AutoAtten_PVT',
        style='pytorch',
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=80,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        agent_sr_ratios='1111',
        num_stages=4,
        # downstream_agent_shapes=[(12, 12), (16, 16), (28, 28), (28, 28)],
        kernel_size=3,
        attn_type='AAAA',
        scale=-0.5,
        atten_cfg=atten_cfg,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/AutoAtten_pvt_small_max_acc.pth')
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150))


gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
