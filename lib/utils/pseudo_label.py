import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from lib.utils.grad_cam import GradCAM
from lib.network.backbone import choose_backbone


# def reshape_transform(tensor, height=14, width=14):
#     #(batch_size, num_patches, hidden_dim). 
#     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

def reshape_transform(tensor):
    batch_size, num_tokens, hidden_dim = tensor.shape

    # Exclude class token
    num_patches = num_tokens - 1

    # Compute spatial size (assuming square patches)
    height = width = int(num_patches ** 0.5)

    if height * width != num_patches:
        raise ValueError(f"Invalid num_patches {num_patches}, cannot reshape.")

    result = tensor[:, 1:, :].reshape(batch_size, height, width, hidden_dim)

    # Change from (B, H, W, C) to (B, C, H, W) for CNN-like processing
    result = result.permute(0, 3, 1, 2)
    return result


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def get_cam_for_image(image_tensor, model, target_layers, target_category=None):
    """
    Args:
        image_tensor: Input image tensor [1, C, H, W]
        model: Classification model
        target_layers: List of target layers for CAM generation
        target_category: Target category for CAM (None for predicted class)

    Returns:
        cam_image: CAM visualization
        grayscale_cam: Grayscale CAM
        pred_class: Predicted class
    """
    model.eval()
    # Create GradCAM object
    # print([layer.__class__.__name__ for layer in target_layers])

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # Get model prediction
    with torch.no_grad():
        image_tensor.requires_grad = True
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]

    # Get predicted class if target_category is None
    if target_category is None:
        pred_class = torch.argmax(output).item()
        target_category = pred_class
    else:
        pred_class = target_category

    print(f"target_category: {target_category}")
    # Generate CAM
    grayscale_cam = cam(image_tensor, [ClassifierOutputTarget(target_category)])
    grayscale_cam = grayscale_cam[0, :]  # Get CAM for first image in batch

    # Convert image tensor to numpy for visualization
    image_np = image_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)

    # Normalize image for visualization
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # Combine image and heatmap
    cam_image = heatmap + np.float32(image_np)
    cam_image = cam_image / np.max(cam_image)

    return cam_image, grayscale_cam, pred_class


def generate_cam_for_dataset(dataloader, model, target_layers, save_dir=None, num_samples_per_batch=3, max_batches=50):
    print("Generating CAM for dataset...")
    device = next(model.parameters()).device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_count = 0

    for batch_idx, (images, labels, image_ids, masks) in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break  # Stop if max_batches limit is reached

        # Move images to the device
        images = images.to(device)
        # Limit to num_samples_per_batch
        num_samples = min(num_samples_per_batch, len(images))
        images = images[:num_samples]
        masks = masks[:num_samples]
        # Get model predictions

        # Prepare figure (4 columns now: Original, CAM, Grayscale CAM, Mask)
        fig, axs = plt.subplots(num_samples, 2, figsize=(20, 5 * num_samples))

        for i, (image, mask, label) in enumerate(zip(images, masks, labels)):
            image_tensor = image.unsqueeze(0)  # Add batch dimension
            cam_image, grayscale_cam, pred_class = get_cam_for_image(image_tensor, model, target_layers)

            # Convert image to NumPy for display
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize

            # Convert mask to NumPy (assuming mask is [B, H, W] or [B, 1, H, W])
            mask_np = mask[i].cpu().numpy()
            if mask_np.ndim == 3:  # [1, H, W]
                mask_np = mask_np.squeeze(0)
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0  # Normalize if needed

            # Plot original image
            axs[i, 0].imshow(image_np)
            axs[i, 0].set_title(f"Original (Label: {label.item()})")
            axs[i, 0].axis('off')

            # Plot CAM
            axs[i, 1].imshow(cam_image)
            axs[i, 1].set_title(f"CAM (Pred: {pred_class})")
            axs[i, 1].axis('off')

            # # Plot grayscale CAM
            # axs[i, 2].imshow(grayscale_cam, cmap='gray')
            # axs[i, 2].set_title("Grayscale CAM")
            # axs[i, 2].axis('off')

            # # Plot ground truth mask
            # axs[i, 3].imshow(mask_np, cmap='gray')
            # axs[i, 3].set_title("Ground Truth Mask")
            # axs[i, 3].axis('off')

        plt.tight_layout()

        # Save or show results
        if save_dir:
            plt.savefig(f"{save_dir}/cam_batch_{batch_idx}.png")
            plt.close()
        else:
            plt.show()

        batch_count += 1


def generate_pseudo_labels(dataloader, model, target_layers, save_dir, threshold):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode

    # Create GradCAM object
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)

    iou_sum = 0.0
    total_samples = 0

    max_batch = 2
    for batch_idx, (images, labels, image_ids, masks) in enumerate(dataloader):
        if max_batch is not None and batch_idx >= max_batch:
            break
        # Move entire batch to device
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle tuple output
                outputs = outputs[0]
            pred_classes = torch.argmax(outputs, dim=1)

        # Process each image in the batch
        for img_idx, (image, pred_class, image_id, mask) in enumerate(zip(images, pred_classes, image_ids, masks)):
            # Generate CAM for predicted class
            targets = [ClassifierOutputTarget(pred_class)]

            # Unsqueeze to add batch dimension
            grayscale_cam = cam(input_tensor=image.unsqueeze(0),
                                targets=targets)

            # Remove batch dimension and ensure 2D array
            grayscale_cam = grayscale_cam[0]  # Shape (H, W)

            # Create binary mask
            pseudo_label = (grayscale_cam > threshold).astype(np.float32)

            # Process mask
            gt_mask = mask.squeeze().cpu().numpy()
            gt_mask = (gt_mask > 0.5).astype(np.float32)  # Ensure binary mask
            intersection = np.logical_and(gt_mask, pseudo_label).sum()
            union = np.logical_or(gt_mask, pseudo_label).sum()
            iou = intersection / (union + 1e-8)  # Avoid division by zero
            iou_sum += iou
            total_samples += 1

            if isinstance(image_id, torch.Tensor):
                img_id = image_id.item()  # Convert tensor to Python scalar
            elif isinstance(image_id, str) and image_id.isdigit():
                img_id = int(image_id)  # Convert string number to int
            else:

                img_id = image_id

            mean = np.array([0.485, 0.456, 0.406])  # Adjust based on your dataset
            std = np.array([0.229, 0.224, 0.225])  # Adjust based on your dataset

            img_np = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            img_np = std * img_np + mean  # Denormalize
            img_np = np.clip(img_np, 0, 1)
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            # Original image

            ax[0].imshow(img_np)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # CAM visualization
            ax[1].imshow(grayscale_cam, cmap='jet')
            ax[1].set_title('Class Activation Map')
            ax[1].axis('off')

            # Pseudo mask
            ax[2].imshow(pseudo_label, cmap='gray')
            ax[2].set_title(f'Pseudo Mask (IoU: {iou:.2f})')
            ax[2].axis('off')

            # Ground truth mask
            ax[3].imshow(gt_mask, cmap='gray')
            ax[3].set_title('Ground Truth Mask')
            ax[3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'visualization_{img_id}.png'), bbox_inches='tight')
            plt.close()

            # Save pseudo-label
            cv2.imwrite(
                os.path.join(save_dir, f"pseudo_label_{img_id}.png"),
                (pseudo_label * 255).astype(np.uint8)
            )

            # Save CAM visualization
            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            cv2.imwrite(
                os.path.join(save_dir, f"cam_{img_id}.png"),
                heatmap
            )

        print(f"Batch {batch_idx + 1}/{len(dataloader)} - Current Mean IoU: {iou_sum / total_samples:.4f}")

    # Final statistics
    print(f"\nFinal Mean IoU: {iou_sum / total_samples:.4f} (over {total_samples} samples)")