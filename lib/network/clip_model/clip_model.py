import torch
import clip
from PIL import Image
import json
import numpy as np
import os

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

max_samples = min(10, len(samples))
for i in range(max_samples):
    sample = samples[i]
    image_path = sample["image"]

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo contains smoke", "a photo contains no smoke"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(f"smoke Sample {i+1} Label probs:", probs)

max_samples = min(10, len(non_smoke))
for i in range(max_samples):
    sample = non_smoke[i]
    image_path = sample["image"]

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo contains smoke", "a photo contains no smoke"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(f"non_smoke Sample {i+1} Label probs:", probs)

