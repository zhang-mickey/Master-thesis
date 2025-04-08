import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import importlib
import os, argparse
import torch.nn.functional as F
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
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.utils.augmentation import *
from lib.utils.image_mask_visualize import *


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    # dataset
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,
                                                                      "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
                        help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
                        help="Path to the image dataset folder")

    parser.add_argument("--non_smoke_image_folder", type=str, default=os.path.join(project_root, "lib/dataset/frames/"),
                        help="Path to the non-smoke image dataset folder")
    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification.pth"),
                        help="Path to save the trained model")
    parser.add_argument("--save_pseudo_labels_path", type=str, default=os.path.join(project_root, "data/pseudo_labels"),
                        help="Path to save the pseudo labels")
    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root, "result/cam"),
                        help="Path to save the cam")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=True, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")

    # train
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--CAM_type", type=str, default='grad',
                        choices=['grad', 'TransCAM', 'Tscam'],
                        help="loss type (default: False)")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--backbone", type=str, default="transformer",
                        help="choose backone")

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()
    print(vars(args))

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # %%
    # image_transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    image_transform, mask_transform = get_transforms(args.img_size)

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

    print(f"Smoke Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    non_smoke_train, non_smoke_val, non_smoke_test = split_non_smoke_dataset(args.non_smoke_image_folder)
    print(
        f"Non-smoke Dataset split: Train={len(non_smoke_train)}, Val={len(non_smoke_val)}, Test={len(non_smoke_test)}")

    # Load smoke images and extract their masks.
    train_smoke = SmokeDataset(
        args.json_path, args.image_folder,
        args.smoke5k, args.Rise,
        transform=image_transform, mask_transform=mask_transform,
        image_ids=train_ids)

    # Load non-smoke images.
    # Select random smoke masks and paste them onto non-smoke images.
    # Ensure realistic blending so that pasted smoke integrates well with the background.
    # Return the modified images and their new mask labels.
    # smoke_aug = SmokeCopyPaste(train_smoke, p=0.7)

    train_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder,
                                       transform=image_transform, mask_transform=mask_transform, image_ids=train_ids,
                                       non_smoke_image_folder=args.non_smoke_image_folder,
                                       non_smoke_files=non_smoke_train,
                                       smoke_dataset=train_smoke,
                                       flag=True)

    random_indices = random.sample(range(len(train_dataset)), 10)

    # Visualize each
    for idx in random_indices:
        show_image_mask_class(train_dataset, idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Training Dataset loaded: {len(train_dataset)} images in total")
    print(f"Number of batches: {len(train_loader)}")

    val_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder,
                                     transform=image_transform,
                                     mask_transform=mask_transform,
                                     image_ids=val_ids,
                                     non_smoke_image_folder=args.non_smoke_image_folder,
                                     non_smoke_files=non_smoke_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Validation Dataset loaded: {len(val_dataset)} images in total")

    test_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder,
                                      transform=image_transform,
                                      mask_transform=mask_transform,
                                      image_ids=test_ids,
                                      non_smoke_image_folder=args.non_smoke_image_folder,
                                      non_smoke_files=non_smoke_test)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test Dataset loaded: {len(test_dataset)} images in total")

    model = choose_backbone(args.backbone)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter = AverageMeter('loss', 'accuracy')
    # max_batches = 1

    if args.CAM_type == 'grad':
        model.train()
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()

            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
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
        model.cuda()
        # Test phase
        print("Starting test phase...")
        test_loss = 0.0
        test_accuracy = 0.0
        test_predictions = []
        test_ground_truth = []

        with torch.no_grad():
            for batch_idx, (images, labels, _, mask) in enumerate(test_loader):
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
                    print(
                        f"Test Batch [{batch_idx + 1}/{len(test_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

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
    elif args.CAM_type == 'TransCAM':
        model.train()

        for epoch in range(1, (args.num_epochs + 1)):
            avg_meter.reset()
            for batch_idx, (img, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                logits_conv, logit_trans, cams = model(img)
                # Combine both logits for final prediction
                combined_logits = logits_conv + logit_trans

                loss = criterion(combined_logits.squeeze(1), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = calculate_accuracy(combined_logits, labels)
                avg_meter.add({'loss': loss.item(), 'accuracy': acc})

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch}/{args.num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

        save_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"{args.backbone}_{os.path.basename(args.save_model_path)}"
        )
        torch.save(model.state_dict(), save_path)
        print("Training complete! Model saved.")
        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.cuda()
        with torch.no_grad():
            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):

                images, labels = images.to(device), labels.float().to(device)
                logits_conv, logit_trans, cams = model(images)
                # for i in range(images.size(0)):
                #     cam = F.interpolate(cam[:, 1:, :, :], orig_img_size, mode='bilinear', align_corners=False)[0]
                #     cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()





