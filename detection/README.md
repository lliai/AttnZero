# Applying AutoAtten to Object Detection

## Data preparation

Prepare COCO according to the guidelines in MMDection and put it in `./data`.



## Training
To train PVT-Tiny + Mask_RCNN on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
./dist_train.sh configs/AutoAtten-PVT/mask_rcnn_autoatten_pvt_t_fpn_1x_coco.py  8
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/AutoAtten-PVT/mask_rcnn_autoatten_pvt_t_fpn_1x_coco.py
```

