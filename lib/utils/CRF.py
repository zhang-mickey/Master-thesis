import numpy as np
from skimage.segmentation import slic, mark_boundaries

try:
    # Try the new location first
    from skimage import graph
except ImportError:
    # Fall back to the old location for backward compatibility
    from skimage.future import graph
from skimage.measure import regionprops
import torch


def apply_crf(image, prob_map, n_segments=100, compactness=10):
    """
    Apply a simplified CRF-like refinement using SLIC superpixels and graph-based segmentation

    Args:
        image: Original image (numpy array) with shape (H, W, 3)
        prob_map: Probability map from model output with shape (H, W)

        n_segments: Number of segments for SLIC(higher = more, smaller superpixels)
        compactness: Compactness parameter for SLIC Higher = more square/regular shaped superpixels

    Returns:
        Refined binary segmentation mask
    """
    # Ensure image is in the right format (0-255 uint8)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Generate superpixels
    segments = slic(image, n_segments=n_segments, compactness=compactness)

    # Create a region adjacency graph
    g = graph.rag_mean_color(image, segments)

    # For each superpixel, calculate the average probability
    refined_mask = np.zeros_like(prob_map)
    for region in regionprops(segments + 1):  # +1 because regionprops requires labels starting from 1
        # Get the coordinates of pixels in this region
        coords = region.coords

        # Calculate average probability in this region
        avg_prob = np.mean(prob_map[coords[:, 0], coords[:, 1]])

        # Assign binary label based on threshold (0.5)
        label = 1 if avg_prob > 0.5 else 0

        # Assign the label to all pixels in the region
        refined_mask[coords[:, 0], coords[:, 1]] = label

    return refined_mask