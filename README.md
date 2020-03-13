# SSD: Single Shot MultiBox Detectorar

This storage is implemented by SSD paper.
Paper backbone network (VGG) and custom backbone network (ResNet) are implemented.

## Follow the steps below for easy use.

## Step 01

```
git clone https://github.com/modernkim/ResNet_SSD_Pytorch.git

cd ResNet_SSD_Pytorch

cd data && mkdir VOCdataset && cd VOCdataset && mkdir VOC_train_val && mkdir VOC_test

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar -C ./VOC_trian_val/
tar xvf VOCtrainval_11-May-2012.tar -C ./VOC_trian_val/
tar xvf VOCtest_06-Nov-2007.tar -C ./VOC_test/
```

## Step 02

```
cd ../../ && mkdir weights && cd weights && mkdir torchvision_pretrained_resnet_rmfc && cd ../

# If you want to use resnet(backbone),

cd load_pretrained_resnet_weight

# and next your choice

python resnet_reducefc.py --choice resnet18 / python3 resnet_reducefc.py resnet18
python resnet_reducefc.py --choice resnet34 / python3 resnet_reducefc.py resnet34
python resnet_reducefc.py --choice resnet50 / python3 resnet_reducefc.py resnet50
python resnet_reducefc.py --choice resnet101 / python3 resnet_reducefc.py resnet101
-----------------------------------------------------------------------------------
# If you want to use vgg(backbone),

cd weights && mkdir vgg_pretrained_reducefc && cd vgg_pretrained_rmfc && cd vgg_pretrained_reducefc
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

## Step 03

```
cd ../../ && vim config.py

### next Modify config.py for your condition.

python train.py / python3 train.py
```
## 

# Result (this repo)

| backbone + SSD |map|
|------|---|
|**paper_vgg**|77.2|
|**reop_vgg** |77.xxx|

## 

| backbone + SSD |map|
|------|---|
|**resnet18**|70.4|
|**resnet34**|73.8|
|**resnet50**|76.2|
|**resnet101**|78.2|


## I referred to the git of the people below. Thank you all.

### overall part --> (HosinPrime / https://github.com/HosinPrime/simple-ssd-for-beginners)

### ./lib/scheduler.py --> (seominseok0429 / https://github.com/seominseok0429/pytorch-warmup-cosine-lr)
