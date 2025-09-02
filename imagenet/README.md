# AttnZero: Efficient Attention Discovery for Vision Transformers

This folder contains the implementation of Atten-Zero based on DeiT, PVT, Swin and CSwin models for ImageNet classification.


## News

- To train `AttnZero-AutoFormer-T` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg cfgs/autoformer/AttnZero_Trial-105_autoformer_t.yaml --data-path <imagenet-path> --output output/Trial-105-AF
```


## Train Models from Scratch

- To train `AttnZero-DeiT/AttnZero-PVT/AttnZero-Swin` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg cfgs/deit/AttnZero_Trial-105_deit_t.yaml --data-path <imagenet-path> --output output/Trial-105
```

- To train `AttnZero-CSwin-T/S/B` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --model-ema --model-ema-decay 0.99984/0.99984/0.99992
```

- To train `AttnZero-Trial-105-Bias-PVT` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg cfgs/pvt/AttnZero_Trial-105-bias_pvt_t.yaml --data-path <imagenet-path> --output output/Trial-105-PVT-Bias
```

- To train `AttnZero-Trial-105-Bias-DeiT` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg cfgs/deit/AttnZero_Trial-105_deit_t_bias.yaml --data-path <imagenet-path> --output output/Trial-105-DEIT-Bias
```
- To train `AttnZero-Trial-105-Bias-Swin` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg cfgs/swin/AttnZero_Trial-105_swin_t_bias.yaml --data-path <imagenet-path> --output output/Trial-105-SWIN-BIAS
```


## Fine-tuning on higher resolution

- Fine-tune a `AttnZero-Swin-B` model pre-trained on 224x224 resolution to 384x384 resolution:


```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/swin/AttnZero_Trial-105_swin_b_384.yaml --data-path <imagenet-path> --output output/Trial-105 --pretrained <path-to-224x224-pretrained-weights>
```

- Fine-tune a `AttnZero-CSwin-B` model pre-trained on 224x224 resolution to 384x384 resolution:


```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg ./cfgs/cswin/AttnZero_Trial-140_cswin_b_384.yaml --data-path <imagenet-path> --output output/Trial-140 --pretrained <path-to-224x224-pretrained-weights> --model-ema --model-ema-decay 0.9998
```


