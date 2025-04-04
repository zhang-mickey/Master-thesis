import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import sys
from torch.utils.data import Dataset, DataLoader

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from lib.utils.metrics import *
from lib.utils.CRF import *


class InferenceDataset(Dataset):
    def __init__(self, image_folder, image_size=(512, 512)):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                            f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize to model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Open image in RGB mode
        image = self.transform(image)  # Apply transformations
        return image, image_path  # Return both image tensor and path


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    # Dataset
    parser.add_argument("--image_folder", type=str, default="smoke-segmentation.v5i.coco-segmentation/test/")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--mask_path", type=str, default=os.path.join(project_root, "output/"))
    # Model
    parser.add_argument("--backbone", type=str, default="deeplabv3plus_resnet101")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = choose_backbone(args.backbone)
    if args.backbone == "Segmenter":
        model.load_state_dict(torch.load("model/Segmenter_model_supervised.pth"))
    elif args.backbone == "deeplabv3plus_resnet101":
        model.load_state_dict(torch.load("model/deeplabv3plus_resnet101_model_supervised.pth"))
    else:
        model.load_state_dict(torch.load("model/deeplabv3_resnet50_model_supervised.pth"))
    model.to(device)
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),  # Ensure the size matches the trained model's input size
    ])

    if not os.path.exists(args.mask_path):
        os.makedirs(args.mask_path)
    dataset = InferenceDataset(args.image_folder)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # Process each image in the dataset
    for idx, (image, image_path) in enumerate(data_loader):
        image = image.to(device)

        with torch.no_grad():
            output = model(image)

            prediction = torch.sigmoid(output).squeeze(0).cpu().numpy()
            prob_map = torch.sigmoid(output).squeeze(0).cpu().numpy()[0]  # Shape [H,W]
        # Load original image for CRF
        original_img = cv2.imread(image_path[0])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Resize probability map to original size
        h, w = original_img.shape[:2]
        prob_map_resized = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        # Apply CRF refinement
        refined_mask = apply_crf(original_img, prob_map_resized) * 255

        # Convert prediction to binary mask (threshold at 0.5)
        original_mask = (prob_map > 0.5).astype(np.uint8) * 255
        original_overlay = original_image.copy()
        original_overlay[refined_mask == 255] = [0, 255, 0]
        # Blend images
        alpha = 0.3  # Transparency factor
        blended = cv2.addWeighted(original_img, 1 - alpha, original_overlay, alpha, 0)
        orig_mask_path = os.path.join(args.mask_path, f"overlay_orig_{idx}.png")
        cv2.imwrite(orig_mask_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        overlay = original_image.copy()
        overlay[refined_mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(original_img, 1 - alpha, overlay, alpha, 0)
        crf_overlay_path = os.path.join(args.mask_path, f"overlay_crf_{idx}.png")
        cv2.imwrite(crf_overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    print("Inference completed. Masks and overlays saved in 'output' folder.")