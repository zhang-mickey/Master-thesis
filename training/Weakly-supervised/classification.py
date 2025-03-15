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
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")

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
from PIL import Image
from lib.utils.pseudo_label import *

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,"smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str, default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"), help="Path to the image dataset folder")
    parser.add_argument("--save_model_path", type=str, default=os.path.join(project_root,"model/model_classification.pth"), help="Path to save the trained model")
    parser.add_argument("--save_pseudo_labels_path", type=str, default=os.path.join(project_root,"data/pseudo_labels"), help="Path to save the pseudo labels")
    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root,"result/cam"), help="Path to save the cam")
    parser.add_argument("--batch_size", type=int, default=8,help="training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="epoch number")
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")
    parser.add_argument("--backbone", type=str, default="transformer", help="choose backone")
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder, transform=image_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = choose_backbone(args.backbone)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter=AverageMeter('loss', 'accuracy')
    max_batches = 1
    for epoch in range(1,(args.num_epochs+1)):
        model.train()
        avg_meter.pop()

        for  batch_idx,(images, labels) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break  # Stop after two batches
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            outputs = outputs.squeeze(1)
            acc = calculate_accuracy(outputs, labels)

            loss = criterion(outputs, labels)



            loss.backward()
            optimizer.step()

            # Update AverageMeter with loss and accuracy
            avg_meter.add({'loss': loss.item(), 'accuracy': acc})

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        # At the end of the epoch, print the average loss and accuracy
        avg_loss, avg_acc = avg_meter.get('loss', 'accuracy')
        print(f"Epoch [{epoch}/{args.num_epochs}], Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%")


    save_path = os.path.join(
    os.path.dirname(args.save_model_path),
    f"{args.backbone}_{os.path.basename(args.save_model_path)}"
)
    torch.save(model.state_dict(), save_path)
    print("Training complete! Model saved.")


    model.load_state_dict(torch.load(save_path))
    model.eval()
    # Determine target layers for CAM   
    if args.backbone == "resnet101":
        target_layers = [model.layer4[-1]]  # Last layer of layer4
    elif args.backbone == "transformer":
        target_layers = [model.blocks[-1].norm1]  # Last transformer block
    else:
        # Adjust for your specific model architecture
        target_layers = [list(model.children())[-3]]  # Example fallback

    
        # Generate and visualize CAMs
    generate_cam_for_dataset(
        dataloader=train_loader,
        model=model,
        target_layers=target_layers,
        save_dir=args.save_cam_path,
    )
    
    # Generate pseudo-labels
    generate_pseudo_labels(
        dataloader=train_loader,
        model=model,
        target_layers=target_layers,
        save_dir=args.save_pseudo_labels_path,
        threshold=0.2
    )