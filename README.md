# Master-thesis

Industrial exhaust smoke emission semantic segmentation using image level annotations.

scarcity of labels

Run on SURF

all the codes are for binary classification and semantic segmentation

## Background

Scarcity of pixel-level annotations.

![image](https://github.com/user-attachments/assets/6aab2ab6-1e8d-47c0-b329-1fd123cd49c7)

Sparse object coverage

inaccurate boundries

spurious corelations

# Prerequisite


## build python extension module
The implementation of DenseCRF loss depends on fast bilateral filtering, which is provided in C++. Use SWIG to wrap C++ for python and then build the python module of bilateral filtering.
```
cd wrapper
swig -python -c++ bilateralfilter.i
python setup.py install
```
# Usage(Multi-stage)

**1. Train a classification network to get CAMs**

**2. Refinement**

**3. Segmentation**



# Dataset

**Train**

IJmond video frames.
[900,900]
Smoke vs non Smoke to train a classifier.

**Test**

900 labeled images [1920,1280]. Instead of resize, crop the images to generate [512,512]

Part of the following images served as out of distribution data

**Smoke 5K**
https://github.com/MultiX-Amsterdam/ijmond-camera-ai/tree/main/bvm_training/trans_bvm

**deep-smoke-machine**
https://github.com/CMU-CREATE-Lab/deep-smoke-machine

# model

download the pretrained model to initiate the model.

```
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
```

## Supervised learning
DeeplabV3+

## multi-stage Weakly supervised learning 
![image](https://github.com/user-attachments/assets/d566e36b-8010-4805-a35c-ddb1940b28b5)

### classifier
ViT

Resnet101

Resnet50

### Refinement

#### optimizing the feature representation for classifier


<img width="1305" alt="image" src="https://github.com/user-attachments/assets/62eef665-af05-45a2-bb75-4a13913339e4" />


##### Consistency Learning

**Boundry-Aware Spurious corelations remove ** 

**Motivation**
CNN-based:better location 

Transformer-based :more clear boundary

<img width="908" alt="image" src="https://github.com/user-attachments/assets/077cacbd-1f8d-48ca-b895-ca458b342d4c" />

ResNet-based architecture as the teacher

Transformer-based architecture as the student model

dual-model architecture using Vision Transformer (ViT) and ResNet50 backbones

#### optimizing CAM




#### post-processing

CRF

SAM-enhanced

CAM-fusion

Multi-scale

Sliding window

CLip-aided

AffinityNet



#####
# Choosing the layer(s) to extract activations from

# Results

As you can see, the refined cam can locate the smoke precisely and no wrongly activated regions(co-occuring pattern) occurs.
<img width="427" alt="image" src="https://github.com/user-attachments/assets/b73cc7d3-0c68-4d73-9bfb-1fdf7e765bbe" />
