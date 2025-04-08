import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch

def denormalize(tensor, mean, std):
    """
    tensor: (C, H, W)
    mean, std: lists of length 3 (RGB)
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)  # Clamp to [0,1] range

def show_image_mask(dataset, idx, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image, mask, image_id = dataset[idx]

    # Denormalize
    image = denormalize(image, mean, std)

    # Convert to PIL image for display
    image = TF.to_pil_image(image.cpu())

    # Convert mask if needed
    if isinstance(mask, torch.Tensor):
        mask = TF.to_pil_image(mask.cpu().squeeze(0))  # (1, H, W) -> (H, W)

    # Show
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image: {image_id}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5,cmap='Reds')
    plt.axis('off')

    plt.show()

def show_image_mask_class(dataset, idx,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image, label, image_id,mask = dataset[idx]

    # Denormalize
    image = denormalize(image, mean, std)

    # Convert to PIL image for display
    image = TF.to_pil_image(image.cpu())

    if isinstance(mask, torch.Tensor):
        mask = TF.to_pil_image(mask.squeeze())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image [{image_id}]")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    plt.show()