_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # pretrained='pretrained/agent_pvt_t_max_acc.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth',
    backbone=dict(
        type='AgentPVT',
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
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        agent_sr_ratios='1111',
        num_stages=4,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes=[(12, 12), (16, 16), (28, 28), (28, 28)],
        kernel_size=3,
        attn_type='AAAA',
        scale=-0.5,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/agent_pvt_t.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
