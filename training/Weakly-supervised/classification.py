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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
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
from PIL import Image
from lib.utils.pseudo_label import *

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,"smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str, default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"), help="Path to the image dataset folder")
    parser.add_argument("--non_smoke_image_folder", type=str, default=os.path.join(project_root, "lib/dataset/frames/"), help="Path to the non-smoke image dataset folder")
    parser.add_argument("--save_model_path", type=str, default=os.path.join(project_root,"model/model_classification.pth"), help="Path to save the trained model")
    parser.add_argument("--save_pseudo_labels_path", type=str, default=os.path.join(project_root,"data/pseudo_labels"), help="Path to save the pseudo labels")
    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root,"result/cam"), help="Path to save the cam")
    parser.add_argument("--batch_size", type=int, default=8,help="training batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="epoch number")
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")
    parser.add_argument("--backbone", type=str, default="transformer", help="choose backone")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')   
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

        #set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

    print(f"Smoke Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    non_smoke_train, non_smoke_val, non_smoke_test = split_non_smoke_dataset(args.non_smoke_image_folder)
    print(f"Non-smoke Dataset split: Train={len(non_smoke_train)}, Val={len(non_smoke_val)}, Test={len(non_smoke_test)}")
    train_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder,
                                     transform=image_transform,image_ids=train_ids,
                                     non_smoke_image_folder=args.non_smoke_image_folder,
                                     non_smoke_files=non_smoke_train)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Training Dataset loaded: {len(train_dataset)} images in total")
    print(f"Number of batches: {len(train_loader)}")
    val_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder, 
                                    transform=image_transform,image_ids=val_ids,
                                    non_smoke_image_folder=args.non_smoke_image_folder,
                                    non_smoke_files=non_smoke_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Validation Dataset loaded: {len(val_dataset)} images in total")

    test_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder, 
                                    transform=image_transform, image_ids=test_ids, 
                                    non_smoke_image_folder=args.non_smoke_image_folder,
                                    non_smoke_files=non_smoke_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test Dataset loaded: {len(test_dataset)} images in total")
    
    model = choose_backbone(args.backbone)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter=AverageMeter('loss', 'accuracy')
    # max_batches = 1
    for epoch in range(1,(args.num_epochs+1)):
        model.train()
        avg_meter.pop()

        for  batch_idx,(images, labels,_) in enumerate(train_loader):
            # if batch_idx >= max_batches:
            #     break  # Stop after two batches
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
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
    #Test phase
    print("Starting test phase...")
    test_loss = 0.0
    test_accuracy = 0.0
    test_predictions = []
    test_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, (images, labels,_) in enumerate(test_loader):
            images, labels = images.to(device), labels.float().to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.squeeze(1)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            
            test_loss += loss.item()
            test_accuracy += acc
            
            # Store predictions and ground truth for metrics calculation
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            test_predictions.extend(predictions.cpu().numpy())
            test_ground_truth.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Test Batch [{batch_idx + 1}/{len(test_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")
    
    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_accuracy / len(test_loader)
    # Calculate additional metrics

    precision = precision_score(test_ground_truth, test_predictions, zero_division=0)
    recall = recall_score(test_ground_truth, test_predictions, zero_division=0)
    f1 = f1_score(test_ground_truth, test_predictions, zero_division=0)
    conf_matrix = confusion_matrix(test_ground_truth, test_predictions)
    
    print("\n===== Test Results =====")
    print(f"Average Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {avg_test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
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
        threshold=args.threshold
    )