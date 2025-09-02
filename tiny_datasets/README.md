## train one network：

```Python
python run_net.py  --mode train  --cfg configs/autoformer-linear_c100_base.yaml
```

## Search valid attention：

```Python
python travel_search_space.py
```


## multi-trial

### 1. prepare ground-truth

```Python
python prepare_gt.py  --refer_cfg configs/autoformer-linear_c100_base.yaml  --info_path candidate.pth  --mode prepare
```

### 2. run sh

```Python
sh work_dirs/GT/configs/autoformer-linear_c100_base/run_bash_0.sh
```

### 3. save multi-trial results 

```Python
python prepare_gt.py  --refer_cfg configs/autoformer-linear_c100_base.yaml  --info_path candidate.pth  --mode save
```



