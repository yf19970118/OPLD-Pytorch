## Install

```
# install pytorch 1.1 and torchvision
sudo pip3 install torch==1.1 torchvision

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
sudo python setup.py install --cuda_ext --cpp_ext

# clone Hier-R-CNN
git clone https://github.com/soeaver/Parsing-R-CNN.git

# install other requirements
pip3 install -r requirements.txt

# mask ops
cd OPLD
sh make.sh


## Data and Pre-train weights

  Make sure to put the files as the following structure:

  ```
  ├─data
  │  ├─DOTA
  │  │  ├─dota_1024_200
  │  │  │  ├─trainval_1024
  │  │  │  │  │─images
  │  │  │  │  │─DOTA_trainval_1024.json
  │  │  │  ├─test_1024  
  │  │  │  │  │─images
  │  │  │  │  │─DOTA_test_1024.json
  |
  ├─weights
     ├─resnet50_caffe.pth
     ├─resnet101_caffe.pth
