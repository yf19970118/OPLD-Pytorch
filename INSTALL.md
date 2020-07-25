## Install

```
# install pytorch 1.1 and torchvision
sudo pip3 install torch==1.1 torchvision

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
sudo python setup.py install --cuda_ext --cpp_ext

# clone OPLD
git clone https://github.com/yf19970118/OPLD-Pytorch.git

# install other requirements
pip3 install -r requirements.txt

# mask ops
cd OPLD
sh make.sh


## Data and Pre-train weights

  Make sure to put the files as the following structure:

  ```
├── data
│   ├── DOTA
│   │   ├── dota_1024_200
│   │   │   ├── test_1024
│   │   │   │   ├── DOTA_test_1024.json
│   │   │   │   └── images
│   │   │   └── trainval_1024
│   │   │       ├── DOTA_trainval_1024.json
│   │   │       └── images
│   │   └── dota_1024_200_ms
│   │       ├── test_1024
│   │       │   ├── DOTA_test_1024_ms.json
│   │       │   └── images
│   │       └── trainval_1024
│   │           ├── DOTA_trainval_1024_ms.json
│   │           └── images
│   └── HRSC2016
│       ├── test
│       │   ├── HRSC2016_test.json
│       │   └── images
│       └── trainval
│           ├── HRSC2016_trainval.json
│           └── images
└── weights
    ├── resnet101_caffe.pth
    └── resnet50_caffe.pth
  ```