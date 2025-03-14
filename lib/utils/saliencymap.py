import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import  torchvision.transforms as transforms

from lib.utils.transform import *

def saliency(img, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()
    # transoform input PIL image to torch.Tensor and normalize
    image_transform, mask_transform = get_transforms()
    input=image_transform(img)
    input.unsqueeze_(0)

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    preds = model(input)
    if isinstance(preds, tuple):
        preds = preds[1]  # Extract the main output

    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # apply inverse transform on image
    with torch.no_grad():
        inv_normalize=reverse_transforms()
        input_img = inv_normalize(input[0])
    # plot image and its saleincy map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()