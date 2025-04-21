# Master-thesis
Run on SURF

all the codes are for binary classification and semantic segmentation

# Prerequisite

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

![img_1.png](img_1.png)

7 month 

video frames

**Smoke 5K**
https://github.com/MultiX-Amsterdam/ijmond-camera-ai/tree/main/bvm_training/trans_bvm

# model

## Supervised learning
DeeplabV3+

## Weakly supervised learning manner

### multi-stage

#### classifier
ViT

Resnet101

#### Refinement

##### Adversarial erasing

##### AffinityNet
![img.png](img.png)

#####
# Choosing the layer(s) to extract activations from

