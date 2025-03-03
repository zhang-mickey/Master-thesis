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
from lib.dataset.contrastiveDataset import SmokeContrastiveDataset
from lib.dataset.SmokeDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from lib.utils.metrics import  *

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default="smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json",help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str, default="smoke-segmentation.v5i.coco-segmentation/test/", help="Path to the image dataset folder")
    parser.add_argument("--save_model_path", type=str, default="model/model_constrastive_learning.pth", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=8,help="training batch size")
    parser.add_argument("--num_epochs", type=int, default=2, help="epoch number")
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()

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
    train_dataset = SmokeContrastiveDataset(args.json_path, args.image_folder, transform=image_transform)
    val_dataset = SmokeContrastiveDataset(args.json_path, args.image_folder, transform=image_transform)
    test_dataset = SmokeContrastiveDataset(args.json_path, args.image_folder, transform=image_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # ---- Load DeepLabV3+ Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model =choose_backbone("deeplabv3plus_resnet50")
    model.to(device)

    # ---- Define Loss & Optimizer ----
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---- Training Loop ----
    max_batches = 2
    for epoch in range(1,(args.num_epochs+1)):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0

        for i,(anchor, positive, negative, anchor_label) in enumerate(train_loader):
            if i >= max_batches:
                break  # Stop after two batches

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Forward pass for the anchor, positive, and negative pairs
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            # Calculate contrastive loss between anchor and positive, anchor and negative
            loss = criterion(anchor_out, positive_out, negative_out)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred_classes = anchor_out.argmax(dim=1)  # Shape: [batch_size, height, width]

            # Ensure anchor_label is in the same shape
            if anchor_label.dim() == 1:  # If anchor_label is [batch_size], reshape it
                anchor_label = anchor_label.view(-1, 1, 1).expand(-1, pred_classes.shape[1], pred_classes.shape[2])

            # Compute accuracy
            acc = calculate_accuracy(pred_classes, anchor_label)

            iou = calculate_iou(anchor_out.squeeze(), anchor_label.squeeze())

            train_accuracy += acc.item()
            train_iou += iou.item()
            anchor_similarity = torch.cosine_similarity(anchor_out, positive_out)
            negative_similarity = torch.cosine_similarity(anchor_out, negative_out)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for i, (anchor, positive, negative, anchor_label) in enumerate(train_loader):

                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                optimizer.zero_grad()

                # Forward pass for the anchor, positive, and negative pairs
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                # Calculate contrastive loss between anchor and positive, anchor and negative
                loss = criterion(anchor_out, positive_out, negative_out)

                val_loss += loss.item()
                # Calculate metrics
                pred_classes = anchor_out.argmax(dim=1)  # Shape: [batch_size, height, width]

                # Ensure anchor_label is in the same shape
                if anchor_label.dim() == 1:  # If anchor_label is [batch_size], reshape it
                    anchor_label = anchor_label.view(-1, 1, 1).expand(-1, pred_classes.shape[1], pred_classes.shape[2])

                # Compute accuracy
                acc = calculate_accuracy(pred_classes, anchor_label)

                iou = calculate_iou(anchor_out.squeeze(), anchor_label.squeeze())

                val_accuracy += acc.item()
                val_iou += iou.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(
            f"Epoch {epoch-1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Train IoU: {avg_train_iou:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, Val IoU: {avg_val_iou:.4f}")
        print(
            f"Anchor-Positive Similarity: {anchor_similarity.mean().item():.4f}, Anchor-Negative Similarity: {negative_similarity.mean().item():.4f}")

    torch.save(model.state_dict(), args.save_model_path)
    print("Training complete!")


    # ---- Inference----