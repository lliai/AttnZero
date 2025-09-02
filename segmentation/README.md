# Applying AutoAtten to Semantic Segmentation

## Data preparation

Prepare ADE-20k according to the guidelines in MMSegmentation and put it in `./data`.



## Training
To train AutoAtten-PVT-Tiny + sem_fpn on ADE-20k on a single node with 8 gpus for 12 epochs run:

```
./dist_train.sh  configs/sem_fpn/AutoAtten-PVT/fpn_autoatten_pvt_t_ade20k_40k.py   8
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/AutoAtten-PVT/fpn_autoatten_pvt_t_ade20k_40k.py
```

