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
from lib.dataset.WeaklyDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from inference.inference import *
from lib.utils.metrics import  *
from lib.utils.saliencymap import *

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,"smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str, default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"), help="Path to the image dataset folder")
    parser.add_argument("--save_model_path", type=str, default=os.path.join(project_root,"model/model_full.pth"), help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=8,help="training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="epoch number")
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")
    parser.add_argument("--backbone", type=str, default="deeplabv3plus_resnet50", help="choose backone")
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

    # Load dataset
    train_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder, transform=image_transform,image_ids=train_ids)
    val_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder, transform=image_transform,image_ids=val_ids)
    test_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder, transform=image_transform,image_ids=test_ids)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # ---- Load DeepLabV3+ Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model =choose_backbone(args.backbone)
    model.to(device)

    # ---- Define Loss & Optimizer ----
    criterion =BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter=AverageMeter('loss')
    # ---- Training Loop ----
    # max_batches = 1
    for epoch in range(1,(args.num_epochs+1)):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0

        for i,(images, labels) in enumerate(train_loader):
            # if i >= max_batches:
            #     break  # Stop after two batches

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics
            acc = calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
            iou = calculate_iou(outputs.squeeze(1), masks.squeeze(1))

            train_accuracy += acc.item()
            train_iou += iou.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))
                val_loss += loss.item()

                # Calculate metrics
                acc = calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
                iou = calculate_iou(outputs.squeeze(1), masks.squeeze(1))

                val_accuracy += acc.item()
                val_iou += iou.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(
            f"Epoch {epoch-1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Train IoU: {avg_train_iou:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, Val IoU: {avg_val_iou:.4f}")

        # Update learning rate
        scheduler.step()

    torch.save(model.state_dict(), args.save_model_path)
    print("Training complete!")


    # ---- Inference----

    # ---- Load the trained model for testing ----
    model.load_state_dict(torch.load(args.save_model_path))
    model.eval()

        # ---- Testing Loop ----
    test_loss = 0.0
    test_accuracy = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_f1 = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks.squeeze(1).float())
            
            # Calculate metrics
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs.squeeze(1), masks.squeeze(1)).item()
            test_iou += calculate_iou(outputs.squeeze(1), masks.squeeze(1)).item()
            
            # Additional metrics
            test_dice += calculate_dice(outputs.squeeze(1), masks.squeeze(1)).item()
            test_f1 += calculate_f1(outputs.squeeze(1), masks.squeeze(1)).item()

    # Calculate averages
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_accuracy / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    avg_test_f1 = test_f1 / len(test_loader)

    print("\nTest Results:")
    print(f"Loss: {avg_test_loss:.4f} | Acc: {avg_test_acc:.4f} | IoU: {avg_test_iou:.4f}")
    print(f"Dice: {avg_test_dice:.4f} | F1: {avg_test_f1:.4f}")

    print("Testing complete!")
