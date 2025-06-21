# Master-thesis

Industrial exhaust smoke emission semantic segmentation using image level annotations.

scarcity of labels

Run on SURF

all the codes are for binary classification and semantic segmentation

## Background

![image](https://github.com/user-attachments/assets/6aab2ab6-1e8d-47c0-b329-1fd123cd49c7)


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

Train
IJmond video frames.
Smoke vs non Smoke to train a classifier.

Test
900 labeled images [1920,1280]. Instead of resize, crop the images to generate [512,512]

served as out of distribution data

**Smoke 5K**
https://github.com/MultiX-Amsterdam/ijmond-camera-ai/tree/main/bvm_training/trans_bvm

**deep-smoke-machine**
https://github.com/CMU-CREATE-Lab/deep-smoke-machine

# model

## Supervised learning
DeeplabV3+

## multi-stage Weakly supervised learning 
![image](https://github.com/user-attachments/assets/d566e36b-8010-4805-a35c-ddb1940b28b5)

### classifier
ViT

Resnet101

### Refinement

#### optimizing the feature representation for classifier
<img width="1305" alt="image" src="https://github.com/user-attachments/assets/62eef665-af05-45a2-bb75-4a13913339e4" />



#### post-processing

CRF

SAM-enhanced

CAM-fusion

Multi-scale



#####
# Choosing the layer(s) to extract activations from

# Results
<img width="427" alt="image" src="https://github.com/user-attachments/assets/b73cc7d3-0c68-4d73-9bfb-1fdf7e765bbe" />
