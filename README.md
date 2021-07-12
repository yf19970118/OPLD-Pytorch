# OPLD:Learning Point-guided Localization for Detection in Remote Sensing Images

What to know before use: 
1、I have tested this project, which can be **compiled** and **tested** correctly. The problems in the above process have nothing to do with my code. 
2、The evaluate result is coco. If you use DOTA, you need to convert the format yourself.
3、If training is unusual, pay attention to the relationship between your batch size and learning rate.
4、Many users can use this code normally. I have changed my profession and no longer maintain this project. Please do not open strange issue again.

Official implementation of **Learning Point-guided Localization for Detection in Remote Sensing Images**

In this repository, we release the OPLD code in Pytorch.

- OPLD architecture:
<p align="center"><img width="90%" src="data/OPLD.png" /></p>

- OPLD output on DOTA:
<p align="center"><img width="90%" src="data/output.png" /></p>


## Installation
- 4 x TITAN X GPU
- pytorch1.1
- python3.6.8

Install OPLD following [INSTALL.md](https://github.com/yf19970118/OPLD-Pytorch/blob/master/INSTALL.md).


### ImageNet pretrained weight

- [R-50](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr)
- [R-50-GN](https://drive.google.com/open?id=1LzcVD7aADhXXY32DdtKhaY9hTXaduhlg)
- [R-101](https://drive.google.com/open?id=1k1N1wuklAYuBD8DX229ZEMsp8opjDJNE)

### on DOTA

|        Model       |  LR  |  mAP50 |  FPS | DOWNLOAD |
|--------------------|:----:|:------:|:-------:| :-------:|
| R-101-FPN_MS       |  1x  | 76.43  | 5.2 |[GoogleDrive](https://drive.google.com/file/d/1JB08rfJ9sAAxucjurG0dScDduuQLzmTS/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/136PdOqx4YFpSq2WpvDo2Jw) (4pt9)|

## Training

To train a model with 4 GPUs run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py  --cfg cfgs/DOTA/e2e_OPLD_R-50-FPN_1x.yaml
```


## Evaluation

### multi-gpu coco evaluation,
```
python tools/test_net.py --cfg ckpts/DOTA/e2e_OPLD_R-50-FPN_1x/e2e_OPLD_R-50-FPN_1x.yaml --gpu_id 0,1,2,3
```

### single-gpu coco evaluation,
```
python tools/test_net.py --cfg ckpts/DOTA/e2e_OPLD_R-50-FPN_1x/e2e_OPLD_R-50-FPN_1x.yaml --gpu_id 0
```

## Citation

If you use [DOTA](https://captain-whu.github.io/DOTA/) dataset and find this repo useful, please consider cite.

```
@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
@ARTICLE{9252176,  
  title={Learning Point-guided Localization for Detection in Remote Sensing Images},  
  author={Q. {Song} and F. {Yang} and L. {Yang} and C. {Liu} and M. {Hu} and L. {Xia}},  
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},   
  year={2020},
}
```

## License
OPLD is released under the [MIT license](https://github.com/yf19970118/OPLD-Pytorch/blob/master/LICENSE).


## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[Detectron](https://github.com/facebookresearch/Detectron/tree/master/detectron)

[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
