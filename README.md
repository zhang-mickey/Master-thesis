# Master-thesis
Run on SURF

all the codes are for binary classification and semantic segmentation

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


# Usage(End to End )


# Results 

**Class Activation Map**

# Dataset

900 labeled images [1920,1280]

Instead of resize, crop the images to generate [512,512]



7 month 

video frames

**Smoke 5K**
https://github.com/MultiX-Amsterdam/ijmond-camera-ai/tree/main/bvm_training/trans_bvm

**deep-smoke-machine**
https://github.com/CMU-CREATE-Lab/deep-smoke-machine
# model

## Supervised learning
DeeplabV3+

## Weakly supervised learning manner

### multi-stage

#### classifier
ViT

Resnet101

#### Refinement

**Threshold**

##### Adversarial erasing

##### AffinityNet


#####
# Choosing the layer(s) to extract activations from

