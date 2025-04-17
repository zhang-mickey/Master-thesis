import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import unary_from_softmax


# from https://github.com/liruiwen/TransCAM/blob/main/tool/visualization.py

def dense_crf(probs, img=None, n_classes=1, n_iter=1, scale_factor=1):
    #

    c, h, w = probs.shape
    if img is not None:
        assert (img.shape[1:3] == (h, w))
        img = np.transpose(img, (1, 2, 0)).copy(order='C')

    d = dcrf.DenseCRF2D(w, h, n_classes)
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    # sxy the bigger
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)

    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(n_iter)
    pres = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))

    return pres


def img_denorm(inputs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), mul=True):
    inputs = np.ascontiguousarray(inputs)
    if inputs.ndim == 3:
        inputs[0, :, :] = (inputs[0, :, :] * std[0] + mean[0])
        inputs[1, :, :] = (inputs[1, :, :] * std[1] + mean[1])
        inputs[2, :, :] = (inputs[2, :, :] * std[2] + mean[2])
    elif inputs.ndim == 4:
        n = inputs.shape[0]
        for i in range(n):
            inputs[i, 0, :, :] = (inputs[i, 0, :, :] * std[0] + mean[0])
            inputs[i, 1, :, :] = (inputs[i, 1, :, :] * std[1] + mean[1])
            inputs[i, 2, :, :] = (inputs[i, 2, :, :] * std[2] + mean[2])

    if mul:
        inputs = inputs * 255
        inputs[inputs > 255] = 255
        inputs[inputs < 0] = 0
        inputs = inputs.astype(np.uint8)
    else:
        inputs[inputs > 1] = 1
        inputs[inputs < 0] = 0
    return inputs

# import numpy as np
# from skimage.segmentation import slic, mark_boundaries
# try:
#     # Try the new location first
#     from skimage import graph
# except ImportError:
#     # Fall back to the old location for backward compatibility
#     from skimage.future import graph
# from skimage.measure import regionprops
# import torch

# def apply_crf(image, prob_map, n_segments=100, compactness=10):
#     """
#     Apply a simplified CRF-like refinement using SLIC superpixels and graph-based segmentation

#     Args:
#         image: Original image (numpy array) with shape (H, W, 3)
#         prob_map: Probability map from model output with shape (H, W)

#         n_segments: Number of segments for SLIC(higher = more, smaller superpixels)
#         compactness: Compactness parameter for SLIC Higher = more square/regular shaped superpixels

#     Returns:
#         Refined binary segmentation mask
#     """
#     # Ensure image is in the right format (0-255 uint8)
#     if image.max() <= 1.0:
#         image = (image * 255).astype(np.uint8)

#     # Generate superpixels
#     segments = slic(image, n_segments=n_segments, compactness=compactness)

#     # Create a region adjacency graph
#     g = graph.rag_mean_color(image, segments)

#     # For each superpixel, calculate the average probability
#     refined_mask = np.zeros_like(prob_map)
#     for region in regionprops(segments + 1):  # +1 because regionprops requires labels starting from 1
#         # Get the coordinates of pixels in this region
#         coords = region.coords

#         # Calculate average probability in this region
#         avg_prob = np.mean(prob_map[coords[:, 0], coords[:, 1]])

#         # Assign binary label based on threshold (0.5)
#         label = 1 if avg_prob > 0.5 else 0

#         # Assign the label to all pixels in the region
#         refined_mask[coords[:, 0], coords[:, 1]] = label

#     return refined_mask