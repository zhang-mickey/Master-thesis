import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import os
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.network.backbone import choose_backbone
from lib.utils import *
from lib.network import *
from lib.loss.loss import *




json_path = "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"

image_folder = "smoke-segmentation.v5i.coco-segmentation/test/"

# ---- Load Dataset & DataLoader ----
# Define transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to match model input
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to match image size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Lambda(lambda x: torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0)))  # Convert to binary mask
])

# Load dataset with transformations
dataset = SmokeDataset(json_path, image_folder, transform=image_transform, mask_transform=mask_transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ---- Load DeepLabV3+ Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model =choose_backbone("deeplabv3plus_resnet50")
model.to(device)

# ---- Define Loss & Optimizer ----
criterion = Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- Training Loop ----
num_epochs = 20
# max_batches = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i,(images, masks) in enumerate(dataloader):
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


    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

# Save the entire model (structure + weights)
# torch.save(model, "/Users/jowonkim/Documents/GitHub/Masterthesis/model/model_full.pth")
# Save only model weights
torch.save(model.state_dict(), "model/model_full.pth")
