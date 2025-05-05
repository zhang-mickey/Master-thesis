import torch
import clip
from PIL import Image
import json
import numpy as np
import os
import torch.nn.functional as F


def apply_mask(original_image, mask_pil):
    """Applies a binary mask to an original image (in pixel space)."""
    # Convert mask to numpy and threshold
    mask_np = np.array(mask_pil.convert("L"))  # Grayscale
    mask_np = (mask_np > 128).astype(np.uint8) * 255  # Binary: 0 or 255

    # Apply mask to original image (set non-masked regions to black)
    original_np = np.array(original_image)
    masked_np = original_np * (mask_np[:, :, np.newaxis] // 255)
    return Image.fromarray(masked_np.astype(np.uint8))


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

smoke_image_dir = "smoke-segmentation.v5i.coco-segmentation/cropped_images/"
non_smoke_dir="smoke-segmentation.v5i.coco-segmentation/non_smoke_images/"

smoke_pseudo_dir="final_model/transformer_GradCAM_0.3_10_pseudo_labels/"
non_smoke_pseudo_dir="final_model/transformer_GradCAM_0.3_30_pseudo_labels/"

samples = []
non_smoke=[]

for mask_name in sorted(os.listdir(smoke_pseudo_dir)):
    if not mask_name.lower().endswith('.png') or 'fusion_pseudo_label' not in mask_name.lower(): continue
    original_image_name = mask_name.replace("fusion_pseudo_label_", "", 1)
    original_image_path = os.path.join(smoke_image_dir, original_image_name)
    samples.append({
        "image": original_image_path,
        "mask": os.path.join(smoke_pseudo_dir, mask_name)
    })
print(len(samples))


for mask_name in sorted(os.listdir(non_smoke_pseudo_dir)):
    if not mask_name.lower().endswith('.png') or 'fusion_pseudo_label' not in mask_name.lower(): continue
    original_image_name = mask_name.replace("fusion_pseudo_label_", "", 1)
    original_image_path = os.path.join(smoke_image_dir, original_image_name)

    non_smoke.append({
        "image": original_image_path,
        "mask": os.path.join(non_smoke_pseudo_dir, mask_name)
    })
print(len(non_smoke))

text = clip.tokenize(["smoke particles  in a dark region", "a solid black background with no visible texture"]).to(device)
max_samples = len(samples)
sum_prob=0
smoke_sim=0

for i in range(max_samples):
    sample = samples[i]
    original_image = Image.open(sample["image"]).convert("RGB")
    mask_pil = Image.open(sample["mask"]).convert("L")
    masked_image = apply_mask(original_image, mask_pil)

    image_tensor = preprocess(masked_image).unsqueeze(0).to(device)  # [1, 3, 224, 224]


    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        similarity = F.cosine_similarity(image_features, text_features)
        smoke_sim+=similarity
        sum_prob+=probs
    # print(f"smoke Sample {i+1} Label probs:", probs)
print(f"sum",sum_prob/len(samples))
print("smoke sim",smoke_sim/max_samples)


max_samples =len(non_smoke)

nonsmoke_sum_prob=0
nonsmoke_sim=0
for i in range(max_samples):
    sample = non_smoke[i]
    original_image = Image.open(sample["image"]).convert("RGB")
    mask_pil = Image.open(sample["mask"]).convert("L")
    masked_image = apply_mask(original_image, mask_pil)

    image_tensor = preprocess(masked_image).unsqueeze(0).to(device)  # [1, 3, 224, 224]


    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image_tensor, text)
        similarity = F.cosine_similarity(image_features, text_features)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        nonsmoke_sim+=similarity
        nonsmoke_sum_prob+=probs
    # print(f"non_smoke Sample {i+1} Label probs:", probs)
print(f"nonsmoke sum",nonsmoke_sum_prob/len(non_smoke))
print(f"nonsmoke sim",nonsmoke_sim/max_samples)


# image_path="/Users/jowonkim/Documents/GitHub/Masterthesis/lib/network/clip_model/img_2.png"
# if True:
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#     text = clip.tokenize(["looks like smoke", "do not looks like smoke"]).to(device)
#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#
#         logits_per_image, logits_per_text = model(image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#     print(f" Label probs:", probs)
#
# image_path = "/Users/jowonkim/Documents/GitHub/Masterthesis/lib/network/clip_model/img_3.png"
# if True:
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#     text = clip.tokenize(["looks like smoke", "do not looks like smoke"]).to(device)
#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#
#         logits_per_image, logits_per_text = model(image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#     print(f" Label probs:", probs)