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



# Classification losses from your logs
cls_losses = [
    0.1976, 0.1037, 0.0886, 0.0761, 0.0866,
    0.0480, 0.0590, 0.0557, 0.0632, 0.0312,
    0.0373, 0.0293, 0.0328, 0.0202, 0.0037,
    0.0001, 0.0001, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000
]

# Background suppression losses from your logs
bg_losses = [
    0.0490, 0.0481, 0.0416, 0.0351, 0.0339,
    0.0253, 0.0232, 0.0185, 0.0193, 0.0177,
    0.0115, 0.0125, 0.0086, 0.0088, 0.0084,
    0.0096, 0.0099, 0.0101, 0.0102, 0.0103,
    0.0104, 0.0105, 0.0105, 0.0106, 0.0106,
    0.0107, 0.0107, 0.0107, 0.0107, 0.0107
]

epochs = range(1, len(cls_losses) + 1)

# Plot it
plt.figure(figsize=(12, 6))
plt.plot(epochs, cls_losses, marker='o', color='tab:blue', label='Classification Loss')
plt.plot(epochs, bg_losses, marker='s', color='tab:red', label='Background Suppression Loss')
plt.title("Losses vs Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.xticks(epochs[::2])
plt.axhline(y=0.0, color='r', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()