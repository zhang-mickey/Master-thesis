import torch
import clip
from PIL import Image
import json
import numpy as np
import os


def split_image_into_patches(image_tensor, patch_size=16):
    """
    Args:
        image_tensor: shape [B, C, H, W]
        patch_size: size of each patch
    Returns:
        patches: list of tensors [B, N_patches, C, patch_size, patch_size]
    """
    B, C, H, W = image_tensor.shape
    patches = image_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(2, 0, 1, 3, 4)  # [N_patches, B, C, p, p]

    return patches


def get_clip_patch_scores(model, preprocess, images, prompt="smoke", device="cuda"):
    """
    For each image in batch, score its patches against a text prompt using CLIP.

    Args:
        model: CLIP model (e.g., "ViT-B/32")
        images: tensor [B, C, H, W]
        prompt: text prompt (e.g., "smoke")

    Returns:
        patch_scores: tensor [B, num_patches]
    """
    B, C, H, W = images.shape
    patch_size = 16
    model.eval()

    # Tokenize prompt
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    # Split into patches
    patches = split_image_into_patches(images, patch_size)
    N_patches = patches.shape[0]  # e.g., 256
    print(f"Total patches: {N_patches}")

    # Encode all patches
    patch_scores = []
    for i in range(N_patches):
        patch = patches[i]  # [B, C, p, p]
        patch = patch.to(device)
        with torch.no_grad():
            image_features = model.encode_image(patch)
            logits = (image_features @ text_features.T).softmax(dim=-1)  # Shape: [B, 1]
            patch_scores.append(logits)

    # Stack and reshape to: [B, N_patches]
    patch_scores = torch.cat(patch_scores, dim=1).to(device)  # [B, N_patches]

    return patch_scores


def clip_guidance_loss(model_attention, clip_scores):
    """
    Encourage model attention map to align with CLIP-guided semantic similarity

    Args:
        model_attention: [B, Num_Heads, Num_Patches]
        clip_scores:     [B, Num_Patches]

    Returns:
        loss: KL between softmax-normalized distributions
    """
    B, H, N = model_attention.shape
    assert clip_scores.shape == (B, N), f"Mismatch: {model_attention.shape} vs {clip_scores.shape}"

    # Average over heads → [B, Num_Patches]
    avg_attn = model_attention.mean(dim=1)

    # Normalize both distributions
    avg_attn = F.softmax(avg_attn, dim=-1)
    clip_scores = F.softmax(clip_scores, dim=-1)

    loss = F.kl_div(
        input=torch.log(avg_attn + 1e-8),
        target=clip_scores,
        reduction='batchmean'
    )

    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

image_dir = "smoke-segmentation.v5i.coco-segmentation/cropped_images/"
non_smoke_dir="smoke-segmentation.v5i.coco-segmentation/non_smoke_images/"
samples = []
non_smoke=[]

for img_name in sorted(os.listdir(image_dir)):
    if img_name.startswith('.'): continue
    img_path = os.path.join(image_dir, img_name)
    samples.append({
        'image': img_path,
        'label': 1,
    })

for img_name in sorted(os.listdir(non_smoke_dir)):
    if img_name.startswith('.'): continue
    non_smoke_path = os.path.join(non_smoke_dir, img_name)
    non_smoke.append({
        'image': non_smoke_path,
        'label': 0
    })


#Encode image to get patch embeddings
max_samples = min(20, len(samples))
text_tokens = clip.tokenize(["a photo contains smoke"]).to(device)
text_tokens_1 = clip.tokenize(["a photo without smoke"]).to(device)
with torch.no_grad():
    text_feat = model.encode_text(text_tokens)  # shape: [1, D]
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    text_feat_1 = model.encode_text(text_tokens_1)  # shape: [1, D]
    text_feat_1 = text_feat_1 / text_feat_1.norm(dim=-1, keepdim=True)

for i in range(max_samples):
    with torch.no_grad():
        sample = samples[i]
        image_path = sample["image"]

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # normalize
        similarities = (image_features @ text_feat.T).squeeze()
        similarities_1 = (image_features @ text_feat_1.T).squeeze()
        print("smoke",similarities)
        print("non_smoke",similarities_1)