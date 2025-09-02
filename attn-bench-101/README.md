## train one network：

```Python
python run_net.py --mode train --cfg configs/swin/autoatten_swin_t_win_56_flower.yaml
```



## multi-trial

### 1. prepare ground-truth

```Python
python prepare_gt.py  --refer_cfg configs/autoformer/autoatten_autoformer_t_c100.yaml  --info_path att_candidates_2000.pth
```

### 2. run sh

```Python
sh work_dirs/GT/autoatten_autoformer_t_c100/run_bash_0.sh
```



