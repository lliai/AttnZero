

# AttnZero: Efficient Attention Discovery for Vision Transformers

[![ECCV 2024](https://img.shields.io/badge/ECCV-2024-blue)](https://eccv2024.ecva.net/)
[![Paper](https://img.shields.io/badge/Paper-ECCV2024-green)](https://link.springer.com/chapter/10.1007/978-3-031-72973-7_2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:** Lujun Li, Zimian Wei, Peijie Dong, Wenhan Luo, Wei Xue, Qifeng Liu, Yike Guo

## 📋 Overview

AttnZero is the first framework for automatically discovering efficient attention modules tailored for Vision Transformers (ViTs). Traditional self-attention in ViTs suffers from quadratic computation complexity O(n²), while our approach discovers linear attention alternatives with O(n) complexity without sacrificing performance.



![ECCV2024_AttnZero_poster_page-0001](ECCV2024_AttnZero_poster_page-0001.jpg)



### ✨ Key Features

- **🔍 Automated Attention Discovery**: Leverages evolutionary algorithms to automatically discover optimal linear attention formulations
- **🏗️ Comprehensive Search Space**: Explores six types of computation graphs with advanced activation, normalization, and binary operators
- **🎯 Multi-objective Optimization**: Optimizes across multiple ViT architectures simultaneously for better generalization
- **⚡ Efficient Search Process**: Implements program checking and rejection protocols for rapid candidate filtering
- **📊 Attn-Bench-101**: Provides a benchmark dataset with precomputed performance metrics for 2,000 attention variants

### 

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- CUDA 11.1+
- torchvision

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AttnZero.git
cd AttnZero

# Create a virtual environment (recommended)
python -m venv attnzero_env
source attnzero_env/bin/activate  # On Windows: attnzero_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
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

## 📝 Citation

If you find AttnZero useful in your research, please cite our paper:



```
@inproceedings{li2024attnzero,
  title={AttnZero: Efficient Attention Discovery for Vision Transformers},
  author={Li, Lujun and Wei, Zimian and Dong, Peijie and Luo, Wenhan and Xue, Wei and Liu, Qifeng and Guo, Yike},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={20--37},
  year={2024},
  organization={Springer}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This work builds upon [DeiT](https://github.com/facebookresearch/deit), [PVT](https://github.com/whai362/PVT), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), and [CSwin Transformer](https://github.com/microsoft/CSWin-Transformer)
- We thank the authors of these foundational works for their contributions
- Special thanks to the ECCV 2024 reviewers for their valuable feedback
