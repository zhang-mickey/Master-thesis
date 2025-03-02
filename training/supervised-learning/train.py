import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import os,argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default="smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json",help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str, default="smoke-segmentation.v5i.coco-segmentation/test/", help="Path to the image dataset folder")
    parser.add_argument("--save_model_path", type=str, default="model/model_full.pth", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=8,help="training batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="epoch number")
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()
    print(torch.cuda.is_available())
    # ---- preprocess ----
    # Get transformations
    image_transform, mask_transform = get_transforms(img_size=512)

    # Split dataset
    train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

    print(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    # Load dataset with transformations
    # dataset = SmokeDataset(json_path, image_folder, transform=image_transform, mask_transform=mask_transform)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load dataset
    train_dataset = SmokeDataset(args.json_path, args.image_folder, transform=image_transform, mask_transform=mask_transform)
    val_dataset = SmokeDataset(args.json_path, args.image_folder, transform=image_transform, mask_transform=mask_transform)
    test_dataset = SmokeDataset(args.json_path, args.image_folder, transform=image_transform, mask_transform=mask_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # ---- Load DeepLabV3+ Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model =choose_backbone("deeplabv3plus_resnet50")
    model.to(device)

    # ---- Define Loss & Optimizer ----
    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---- Training Loop ----
    # max_batches = 2
    for epoch in range(1,(args.num_epochs+1)):
        model.train()
        running_loss = 0.0
        for i,(images, masks) in enumerate(train_loader):
            # if i >= max_batches:
            #     break  # Stop after two batches

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks.squeeze(1))  # Remove channel dimension
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")
    # Save the entire model (structure + weights)
    # torch.save(model, "/Users/jowonkim/Documents/GitHub/Masterthesis/model/model_full.pth")
    # Save only model weights

    torch.save(model.state_dict(), args.save_model_path)
    print("Training complete!")