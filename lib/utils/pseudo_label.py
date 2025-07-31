import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from lib.utils.grad_cam import GradCAM
from lib.network.backbone import choose_backbone
from lib.utils.augmentation import *
import torch.nn.functional as F
import math

# def reshape_transform(tensor, height=14, width=14):
#     #(batch_size, num_patches, hidden_dim).
#     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

# reshape the output tensor from a Vision Transformer (ViT) model
# (or similar transformer-based models) so that it can be visualized or processed
# in a way similar to convolutional neural network (CNN) feature maps.
class_list = ["bg", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
              'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'tvmonitor']


def reshape_transform(tensor, height=16, width=16):
    """

    - ViT/DeiT (assumes class token at position 0, square patches)
    - MiT (e.g., SegFormer, no class token, known H×W)
    """
    # For CNN-like feature maps, just return as is
    if tensor.dim() == 4:
        return tensor
    else:
        #  # For ViT-like models
        batch_size, num_tokens, hidden_dim = tensor.shape
        h = w = int(num_tokens ** 0.5)

        if h * w == num_tokens:
            # For MiT (e.g., SegFormer)
            total_elements = tensor.numel()
            spatial_elements = total_elements // (batch_size * hidden_dim)
            height = width = int(math.sqrt(spatial_elements))
            reshaped = tensor.reshape(batch_size, height, width, hidden_dim)
            return reshaped.permute(0, 3, 1, 2)
        # Exclude class token
        else:
            # For ViT/DeiT (assume 1 class token)
            num_patches = num_tokens - 1
            # Compute spatial size (assuming square patches)
            height = width = int(num_patches ** 0.5)

            if height * width != num_patches:
                raise ValueError(f"Cannot infer square patch layout from {num_patches} patches.")

            result = tensor[:, 1:, :].reshape(batch_size, height, width, hidden_dim)

            # Change from (B, H, W, C) to (B, C, H, W) for CNN-like processing
            result = result.permute(0, 3, 1, 2)
            return result


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
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


class SegmentationOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # model_output: [B, C, H, W]
        return model_output[:, self.category, :, :].mean()


def get_cam_for_image(image_tensor, model, target_layers=None, target_category=None):
    model.eval()
    # print([layer.__class__.__name__ for layer in target_layers])
    if model is None:
        raise ValueError("Model is None in get_cam_for_image")

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # with torch.no_grad():
    #     image_tensor.requires_grad = True
    image_tensor = image_tensor.detach().requires_grad_()
    output = model(image_tensor)
    if isinstance(output, tuple):
        output = output[0]

    if target_category is None:
        pred_class = torch.argmax(output).item()
        target_category = pred_class
    else:
        pred_class = target_category

    # print(f"target_category: {target_category}")
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


def generate_PCM_cam(dataloader,
                     model,
                     save_dir=None,
                     aug=False,
                     num_samples_per_batch=3,
                     max_batches=90):
    device = next(model.parameters()).device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_count = 0

    for batch_idx, (images, labels, image_ids, masks) in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        num_samples = min(num_samples_per_batch, len(images))
        images = images[:num_samples]
        masks = masks[:num_samples]

        aug_images = augment_batch(images)

        fig, axs = plt.subplots(num_samples, 2, figsize=(20, 5 * num_samples))

        for i, (image, aug_image, mask, label) in enumerate(zip(images, aug_images, masks, labels)):
            image_tensor = image.unsqueeze(0)  # Add batch dimension
            _, C, H, W = image_tensor.shape
            aug_tensor = aug_image.unsqueeze(0)

            pred, cam_map, _ = model(image_tensor)

            cam_map = F.interpolate(cam_map, (H, W), mode='bilinear')[0]

            pred_class = torch.argmax(pred[0]).item()

            class_activation = cam_map[pred_class].cpu().detach().numpy()

            class_activation = np.maximum(class_activation, 0)
            grayscale_cam = (class_activation - class_activation.min()) / (
                        class_activation.max() - class_activation.min() + 1e-8)

            # print("pred shape:", pred.shape)            # [1, num_classes]
            # print("cam_map shape:", cam_map.shape)      # [C, H, W] after cam_map = cam_map[0]
            # print("pred_class:", pred_class)

            image_np = image_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

            aug_np = aug_image.cpu().numpy().transpose(1, 2, 0)
            aug_np = (aug_np - aug_np.min()) / (aug_np.max() - aug_np.min())

            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255
            cam_image = heatmap + np.float32(image_np)
            cam_image = cam_image / np.max(cam_image)

            # Convert mask to NumPy
            mask_np = mask[i].cpu().numpy()
            if mask_np.ndim == 3:  # [1, H, W]
                mask_np = mask_np.squeeze(0)
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0

                # Plot original image
            axs[i, 0].imshow(image_np)
            axs[i, 0].set_title(f"Original (Label: {label.item()})")
            axs[i, 0].axis('off')

            # Plot CAM
            axs[i, 1].imshow(cam_image)
            axs[i, 1].set_title(f"CAM (Pred: {pred_class})")
            axs[i, 1].axis('off')

            # # Plot aug image
            # axs[i, 2].imshow(aug_np)
            # axs[i, 2].set_title(f"Aug Image (Label: {label.item()})")
            # axs[i, 2].axis('off')

            # # Plot aug_CAM
            # axs[i, 3].imshow(cam_image)
            # axs[i, 3].set_title(f"CAM (Pred: {pred_class_aug})")
            # axs[i, 3].axis('off')

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/cam_batch_{batch_idx}.png")
            plt.close()
        else:
            plt.show()

        batch_count += 1
    return 0


def generate_PCM_pseudo_labels(dataloader, model, save_dir, threshold):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    scales = [0.5, 1.0, 1.5, 2.0]
    # scales = [1.0]
    scale_metrics = {s: {'iou_sum': 0.0, 'count': 0} for s in scales}
    scale_metrics['avg'] = {'iou_sum': 0.0, 'count': 0}

    for batch_idx, (images, labels, image_ids, masks) in enumerate(dataloader):
        B, C, h_orig, w_orig = images.shape
        images = images.to(device)
        # if batch_idx==1:
        if True:
            for j in range(B):
                scale_cams = []
                scale_strs = []
                image = images[j]
                image_id = image_ids[j]
                mask = masks[j]
                for scale in scales:
                    h_new, w_new = int(h_orig * scale), int(w_orig * scale)
                    image_resized = F.interpolate(image.unsqueeze(0), size=(h_new, w_new), mode='bilinear',
                                                  align_corners=False)

                    with torch.no_grad():
                        pred, cam_map, _ = model(image_resized)

                    pred_class = torch.argmax(pred[0]).item()

                    cam_map = F.interpolate(cam_map, (h_orig, w_orig), mode='bilinear')[0]

                    # class_activation = cam_map[pred_class].cpu().detach().numpy()
                    class_activation = cam_map[pred_class].cpu().detach().numpy()  # [H, W]

                    class_activation = np.maximum(class_activation, 0)
                    grayscale_cam = (class_activation - class_activation.min()) / (
                                class_activation.max() - class_activation.min() + 1e-8)

                    if scale != 1.0:  # No need to resize if scale is already 1.0
                        grayscale_cam = cv2.resize(grayscale_cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

                    pseudo_label = (grayscale_cam > threshold).astype(np.float32)

                    gt_mask = mask.squeeze().cpu().numpy()
                    gt_mask = (gt_mask > 0.5).astype(np.float32)

                    intersection = np.logical_and(gt_mask, pseudo_label).sum()
                    union = np.logical_or(gt_mask, pseudo_label).sum()
                    iou = intersection / (union + 1e-8)

                    scale_metrics[scale]['iou_sum'] += iou
                    scale_metrics[scale]['count'] += 1

                    scale_strs.append(str(scale))
                    scale_cams.append(grayscale_cam)

                if isinstance(image_id, torch.Tensor):
                    img_id = image_id.item()
                elif isinstance(image_id, str) and image_id.isdigit():
                    img_id = int(image_id)
                else:
                    img_id = image_id

                if j <= 2:
                    # image_scale_cams = scale_cams[-4:]
                    # image_scale_strs = scale_strs[-4:]
                    fig, axs = plt.subplots(1, len(scales) + 1, figsize=((len(scales) + 1) * 5, 5))
                    fused_cam = np.mean(scale_cams, axis=0)
                    for i in range(len(scales)):
                        axs[i].imshow(scale_cams[i], cmap='jet')
                        axs[i].set_title(f'CAM @ Scale {scale_strs[i]}')
                        axs[i].axis('off')
                    axs[len(scales)].imshow(fused_cam, cmap='jet')
                    axs[len(scales)].set_title(f'fused_CAM')
                    axs[len(scales)].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"all_scales_cam_{image_id}.png"), bbox_inches='tight')
                    plt.close()

                fused_cam = np.mean(scale_cams, axis=0)

                # fused_cam=np.max(scale_cams,axis=0)

                pseudo_label = (fused_cam > threshold).astype(np.float32)

                intersection = np.logical_and(gt_mask, pseudo_label).sum()
                union = np.logical_or(gt_mask, pseudo_label).sum()
                iou = intersection / (union + 1e-8)

                scale_metrics['avg']['iou_sum'] += iou
                scale_metrics['avg']['count'] += 1

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

                img_np = image.cpu().numpy().transpose(1, 2, 0)
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(img_np)
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                ax[1].imshow(fused_cam, cmap='jet')
                ax[1].set_title('Class Activation Map (Avg)')
                ax[1].axis('off')

                ax[2].imshow(pseudo_label, cmap='gray')
                ax[2].set_title(f'Pseudo Mask (IoU: {iou:.2f})')
                ax[2].axis('off')

                ax[3].imshow(gt_mask, cmap='gray')
                ax[3].set_title('Ground Truth Mask')
                ax[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'fusion_visualization_{img_id}.png'), bbox_inches='tight')
                plt.close()

                # Save pseudo-label
                cv2.imwrite(
                    os.path.join(save_dir, f"fusion_pseudo_label_{img_id}.png"),
                    (pseudo_label * 255).astype(np.uint8)
                )
                # Save CAM heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * fused_cam), cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(save_dir, f"fusion_cam_{img_id}.png"),
                    heatmap
                )

                # print(type(grayscale_cam))  # Should be <class 'numpy.ndarray'>
                # print(grayscale_cam.dtype)  # Should be float32
                np.save(
                    os.path.join(save_dir, f"fusion_cam_{img_id}.npy"),
                    fused_cam
                )

    print("\nScale-wise IoU Results:")
    for scale, metrics in scale_metrics.items():
        mean_iou = metrics['iou_sum'] / metrics['count']
        print(f"Scale {scale}: Mean IoU = {mean_iou:.4f}")


def sliding_window_cam(image_tensor, model, target_layers,
                       window_ratio=0.25, stride_ratio=0.125,
                       min_window=64, min_stride=32, return_patches=False):
    _, C, H, W = image_tensor.shape
    window_size = max(int(min(H, W) * window_ratio), min_window)
    stride = max(int(min(H, W) * stride_ratio), min_stride)

    full_cam = np.zeros((H, W))
    count_map = np.zeros((H, W))
    patches = []
    tops = list(range(0, H - window_size + 1, stride))
    if (H - window_size) % stride != 0:
        tops.append(H - window_size)

    lefts = list(range(0, W - window_size + 1, stride))
    if (W - window_size) % stride != 0:
        lefts.append(W - window_size)

    for top in tops:
        for left in lefts:
            patch = image_tensor[:, :, top:top + window_size, left:left + window_size]

            cam_image, grayscale_cam, pred_class = get_cam_for_image(patch, model, target_layers)
            cam_patch = grayscale_cam
            full_cam[top:top + window_size, left:left + window_size] += cam_patch
            count_map[top:top + window_size, left:left + window_size] += 1
            if return_patches:
                patches.append((patch.squeeze(0).cpu(), cam_image))
    full_cam /= np.maximum(count_map, 1e-6)
    full_cam = (full_cam - full_cam.min()) / (full_cam.max() - full_cam.min() + 1e-6)

    if return_patches:
        return full_cam, patches
    else:
        return full_cam


def visualize_sliding_cam_with_patches(image_tensor, cam_image, sliding_patches, label, save_path=None):
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)

    num_patches = len(sliding_patches)
    fig, axs = plt.subplots(num_patches + 1, 2, figsize=(8, 4 * (num_patches + 1)))

    axs[0, 0].imshow(image_np)
    axs[0, 0].set_title(f"Original (Label: {label.item()})")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cam_image)
    axs[0, 1].set_title("Full CAM")
    axs[0, 1].axis('off')

    for i, (patch_tensor, patch_cam) in enumerate(sliding_patches):
        patch_np = patch_tensor.numpy().transpose(1, 2, 0)
        patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-6)

        axs[i + 1, 0].imshow(patch_np)
        axs[i + 1, 0].set_title(f"Patch {i}")
        axs[i + 1, 0].axis('off')

        axs[i + 1, 1].imshow(patch_cam)
        axs[i + 1, 1].set_title(f"Patch {i} CAM")
        axs[i + 1, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def sliding_window_patch_cam_generate(dataloader, model,
                                      target_layers,
                                      save_dir=None,
                                      aug=False,
                                      num_samples_per_batch=3,
                                      max_batches=1):
    device = next(model.parameters()).device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_count = 0

    for batch_idx, (images, labels, image_ids, masks) in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        # Limit to num_samples_per_batch
        num_samples = min(num_samples_per_batch, len(images))
        images = images[:num_samples]
        masks = masks[:num_samples]

        fig, axs = plt.subplots(num_samples, 3, figsize=(30, 5 * num_samples))

        for i, (image, mask, label) in enumerate(zip(images, masks, labels)):
            image_tensor = image.unsqueeze(0)  # Add batch dimension

            cam_image, grayscale_cam, pred_class = get_cam_for_image(image_tensor, model, target_layers)
            sliding_cam, patch_list = sliding_window_cam(image_tensor, model, target_layers, return_patches=True)
            visualize_sliding_cam_with_patches(
                image_tensor=image,
                cam_image=cam_image,
                sliding_patches=patch_list,
                label=label,
                save_path=f"{save_dir}/patch_cam_{batch_idx}_img{i}.png"
            )


def sliding_window_cam_generate(dataloader, model,
                                target_layers,
                                save_dir=None,
                                aug=False,
                                ):
    device = next(model.parameters()).device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    scales = [0.5, 1.0, 1.5, 2.0]
    scale_metrics = {s: {'iou_sum': 0.0, 'count': 0} for s in scales}
    scale_metrics['avg'] = {'iou_sum': 0.0, 'count': 0}

    for batch_idx, (images, labels, image_ids, masks) in enumerate(dataloader):
        B, C, h_orig, w_orig = images.shape
        images = images.to(device)
        for j in range(B):
            scale_cams = []
            scale_strs = []
            image = images[j]
            image_id = image_ids[j]
            mask = masks[j]
            gt_mask = mask.squeeze().cpu().numpy()
            gt_mask = (gt_mask > 0.5).astype(np.float32)

            for scale in scales:
                h_new, w_new = int(h_orig * scale), int(w_orig * scale)
                image_resized = F.interpolate(image.unsqueeze(0), size=(h_new, w_new), mode='bilinear',
                                              align_corners=False)

                sliding_cam = sliding_window_cam(image_resized, model, target_layers)

                if scale != 1.0:
                    sliding_cam = cv2.resize(sliding_cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

                pseudo_label = (sliding_cam > 0.3).astype(np.float32)

                intersection = np.logical_and(gt_mask, pseudo_label).sum()
                union = np.logical_or(gt_mask, pseudo_label).sum()
                iou = intersection / (union + 1e-8)

                scale_metrics[scale]['iou_sum'] += iou
                scale_metrics[scale]['count'] += 1
                scale_strs.append(str(scale))
                scale_cams.append(sliding_cam)

            fused_cam = np.mean(scale_cams, axis=0)
            pseudo_label = (fused_cam > 0.3).astype(np.float32)

            intersection = np.logical_and(gt_mask, pseudo_label).sum()
            union = np.logical_or(gt_mask, pseudo_label).sum()
            iou = intersection / (union + 1e-8)

            scale_metrics['avg']['iou_sum'] += iou
            scale_metrics['avg']['count'] += 1

    print("\nScale-wise IOU:")
    for scale, metrics in scale_metrics.items():
        avg_iou = metrics['iou_sum'] / metrics['count']
        print(f"Scale {scale}: Average IOU = {avg_iou:.4f}")

    #     fig, axs = plt.subplots(num_samples, 3, figsize=(30, 5 * num_samples))

    #     for i, (image,mask,label) in enumerate(zip(images, masks, labels)):
    #         image_tensor = image.unsqueeze(0)  # Add batch dimension

    #         cam_image, grayscale_cam, pred_class = get_cam_for_image(image_tensor, model, target_layers)
    #         sliding_cam = sliding_window_cam(image_tensor, model, target_layers)

    #         # Convert image to NumPy for display
    #         image_np = image.cpu().numpy().transpose(1, 2, 0)
    #         image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize

    #         # Convert mask to NumPy
    #         mask_np = mask[i].cpu().numpy()
    #         if mask_np.ndim == 3:  # [1, H, W]
    #             mask_np = mask_np.squeeze(0)
    #         if mask_np.max() > 1:
    #             mask_np = mask_np / 255.0

    #         # Plot original image
    #         axs[i, 0].imshow(image_np)
    #         axs[i, 0].set_title(f"Original (Label: {label.item()})")
    #         axs[i, 0].axis('off')

    #         # Plot CAM
    #         axs[i, 1].imshow(cam_image)
    #         axs[i, 1].set_title(f"CAM (Pred: {pred_class})")
    #         axs[i, 1].axis('off')

    #         # Plot  sliding window cam
    #         axs[i, 2].imshow(sliding_cam, cmap='jet')
    #         axs[i, 2].set_title("Sliding Window CAM")
    #         axs[i, 2].axis('off')
    #     plt.tight_layout()

    #     if save_dir:
    #         plt.savefig(f"{save_dir}/sw_cam_batch_{batch_idx}.png")
    #         plt.close()
    #     else:
    #         plt.show()

    #     batch_count += 1
    # return 0


def generate_cam_for_dataset(dataloader, model,
                             target_layers=None,
                             save_dir=None,
                             aug=False,
                             num_samples_per_batch=3,
                             max_batches=90):
    device = next(model.parameters()).device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_count = 0

    for batch_idx, (image_ids, images, masks, labels) in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        # Limit to num_samples_per_batch
        num_samples = min(num_samples_per_batch, len(images))
        images = images[:num_samples]
        masks = masks[:num_samples]

        aug_images = augment_batch(images)

        fig, axs = plt.subplots(num_samples, 4, figsize=(40, 5 * num_samples))

        for i, (image, aug_image, mask, label) in enumerate(zip(images, aug_images, masks, labels)):
            image_tensor = image.unsqueeze(0)  # Add batch dimension
            aug_tensor = aug_image.unsqueeze(0)

            cam_image, grayscale_cam, pred_class = get_cam_for_image(image_tensor, model, target_layers)
            cam_image_aug, grayscale_cam_aug, pred_class_aug = get_cam_for_image(aug_tensor, model, target_layers)

            # Convert image to NumPy for display
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize
            aug_np = aug_image.cpu().numpy().transpose(1, 2, 0)
            aug_np = (aug_np - aug_np.min()) / (aug_np.max() - aug_np.min())
            # Convert mask to NumPy
            mask_np = mask[i].cpu().numpy()
            if mask_np.ndim == 3:  # [1, H, W]
                mask_np = mask_np.squeeze(0)
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0

                # Plot original image
            axs[i, 0].imshow(image_np)
            label_indices = (label > 0).nonzero(as_tuple=True)[0].tolist()
            label_names = [class_list[idx] for idx in label_indices]
            axs[i, 0].set_title(f"Original (Labels: {', '.join(label_names)})")
            axs[i, 0].axis('off')

            # Plot CAM
            axs[i, 1].imshow(cam_image)
            axs[i, 1].set_title(f"CAM (Pred: {pred_class})")
            axs[i, 1].axis('off')

            # Plot aug image
            axs[i, 2].imshow(aug_np)
            label_indices = (label > 0).nonzero(as_tuple=True)[0].tolist()
            label_names = [class_list[idx] for idx in label_indices]
            axs[i, 2].set_title(f"Aug (Labels: {', '.join(label_names)})")
            axs[i, 2].axis('off')

            # Plot aug_CAM
            axs[i, 3].imshow(cam_image_aug)
            axs[i, 3].set_title(f"CAM (Pred: {pred_class_aug})")
            axs[i, 3].axis('off')

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/cam_batch_{batch_idx}.png")
            plt.close()
        else:
            plt.show()

        batch_count += 1
    return 0


def generate_crop_cam_for_dataset(dataloader, model,
                                  target_layers,
                                  save_dir=None,
                                  aug=False,
                                  num_samples_per_batch=3,
                                  max_batches=90):
    device = next(model.parameters()).device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_count = 0

    for batch_idx, (image_ids, images, masks, labels) in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        # Limit to num_samples_per_batch
        num_samples = min(num_samples_per_batch, len(images))
        images = images[:num_samples]
        masks = masks[:num_samples]

        aug_images = augment_batch(images)

        fig, axs = plt.subplots(num_samples, 4, figsize=(40, 5 * num_samples))

        for i, (image, aug_image, mask, label) in enumerate(zip(images, aug_images, masks, labels)):
            image_tensor = image.unsqueeze(0)  # Add batch dimension
            aug_tensor = aug_image.unsqueeze(0)

            cam_image, grayscale_cam, pred_class = get_cam_for_image(image_tensor, model, target_layers)
            cam_image_aug, grayscale_cam_aug, pred_class_aug = get_cam_for_image(aug_tensor, model, target_layers)

            # Convert image to NumPy for display
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize
            aug_np = aug_image.cpu().numpy().transpose(1, 2, 0)
            aug_np = (aug_np - aug_np.min()) / (aug_np.max() - aug_np.min())
            # Convert mask to NumPy
            mask_np = mask[i].cpu().numpy()
            if mask_np.ndim == 3:  # [1, H, W]
                mask_np = mask_np.squeeze(0)
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0

                # Plot original image
            axs[i, 0].imshow(image_np)
            label_indices = (label > 0).nonzero(as_tuple=True)[0].tolist()
            label_names = [class_list[idx] for idx in label_indices]
            axs[i, 0].set_title(f"Original (Labels: {', '.join(label_names)})")
            axs[i, 0].axis('off')

            # Plot CAM
            axs[i, 1].imshow(cam_image)
            axs[i, 1].set_title(f"CAM (Pred: {pred_class})")
            axs[i, 1].axis('off')

            # Plot aug image
            axs[i, 2].imshow(aug_np)
            label_indices = (label > 0).nonzero(as_tuple=True)[0].tolist()
            label_names = [class_list[idx] for idx in label_indices]
            axs[i, 2].set_title(f"Aug (Labels: {', '.join(label_names)})")
            axs[i, 2].axis('off')

            # Plot aug_CAM
            axs[i, 3].imshow(cam_image_aug)
            axs[i, 3].set_title(f"CAM (Pred: {pred_class_aug})")
            axs[i, 3].axis('off')

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/cam_batch_{batch_idx}.png")
            plt.close()
        else:
            plt.show()

        batch_count += 1
    return 0


def generate_pseudo_labels(dataloader, model, target_layers, save_dir, threshold):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    scales = [0.5, 1.0, 1.5, 2.0]
    # scales = [1.0]
    scale_metrics = {s: {'iou_sum': 0.0, 'count': 0} for s in scales}
    scale_metrics['avg'] = {'iou_sum': 0.0, 'count': 0}

    for batch_idx, (image_ids, images, masks, labels) in enumerate(dataloader):
        B, C, h_orig, w_orig = images.shape
        images = images.to(device)
        # if batch_idx==1:
        if True:
            for j in range(B):
                scale_cams = []
                scale_strs = []
                image = images[j]
                image_id = image_ids[j]
                mask = masks[j]
                gt_mask = mask.squeeze().cpu().numpy()
                gt_mask = (gt_mask > 0.5).astype(np.float32)
                for scale in scales:
                    h_new, w_new = int(h_orig * scale), int(w_orig * scale)
                    image_resized = F.interpolate(image.unsqueeze(0), size=(h_new, w_new), mode='bilinear',
                                                  align_corners=False)

                    with torch.no_grad():
                        outputs = model(image_resized)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                    pred_class = torch.argmax(outputs, dim=1)

                    grayscale_cam = cam(input_tensor=image_resized, targets=[ClassifierOutputTarget(pred_class)])[0]
                    if scale != 1.0:  # No need to resize if scale is already 1.0
                        grayscale_cam = cv2.resize(grayscale_cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

                    pseudo_label = (grayscale_cam > threshold).astype(np.float32)

                    intersection = np.logical_and(gt_mask, pseudo_label).sum()
                    union = np.logical_or(gt_mask, pseudo_label).sum()
                    iou = intersection / (union + 1e-8)

                    scale_metrics[scale]['iou_sum'] += iou
                    scale_metrics[scale]['count'] += 1
                    scale_strs.append(str(scale))
                    scale_cams.append(grayscale_cam)
                if isinstance(image_id, torch.Tensor):
                    img_id = image_id.item()
                elif isinstance(image_id, str) and image_id.isdigit():
                    img_id = int(image_id)
                else:
                    img_id = image_id

                if j <= 2:
                    # image_scale_cams = scale_cams[-4:]
                    # image_scale_strs = scale_strs[-4:]
                    fig, axs = plt.subplots(1, len(scales) + 1, figsize=((len(scales) + 1) * 5, 5))
                    fused_cam = np.mean(scale_cams, axis=0)
                    for i in range(len(scales)):
                        axs[i].imshow(scale_cams[i], cmap='jet')
                        axs[i].set_title(f'CAM @ Scale {scale_strs[i]}')
                        axs[i].axis('off')
                    axs[len(scales)].imshow(fused_cam, cmap='jet')
                    axs[len(scales)].set_title(f'fused_CAM')
                    axs[len(scales)].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"all_scales_cam_{image_id}.png"), bbox_inches='tight')
                    plt.close()

                fused_cam = np.mean(scale_cams, axis=0)

                # fused_cam=np.max(scale_cams,axis=0)

                pseudo_label = (fused_cam > threshold).astype(np.float32)

                intersection = np.logical_and(gt_mask, pseudo_label).sum()
                union = np.logical_or(gt_mask, pseudo_label).sum()
                iou = intersection / (union + 1e-8)

                scale_metrics['avg']['iou_sum'] += iou
                scale_metrics['avg']['count'] += 1

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])

                img_np = image.cpu().numpy().transpose(1, 2, 0)
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(img_np)
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                ax[1].imshow(fused_cam, cmap='jet')
                ax[1].set_title('Class Activation Map (Avg)')
                ax[1].axis('off')

                ax[2].imshow(pseudo_label, cmap='gray')
                ax[2].set_title(f'Pseudo Mask (IoU: {iou:.2f})')
                ax[2].axis('off')

                ax[3].imshow(gt_mask, cmap='gray')
                ax[3].set_title('Ground Truth Mask')
                ax[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'fusion_visualization_{img_id}.png'), bbox_inches='tight')
                plt.close()

                # Save pseudo-label
                cv2.imwrite(
                    os.path.join(save_dir, f"fusion_pseudo_label_{img_id}.png"),
                    (pseudo_label * 255).astype(np.uint8)
                )
                # Save CAM heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * fused_cam), cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(save_dir, f"fusion_cam_{img_id}.png"),
                    heatmap
                )

                # print(type(grayscale_cam))  # Should be <class 'numpy.ndarray'>
                # print(grayscale_cam.dtype)  # Should be float32
                np.save(
                    os.path.join(save_dir, f"fusion_cam_{img_id}.npy"),
                    fused_cam
                )

    print("\nScale-wise IoU Results:")
    for scale, metrics in scale_metrics.items():
        mean_iou = metrics['iou_sum'] / metrics['count']
        print(f"Scale {scale}: Mean IoU = {mean_iou:.4f}")