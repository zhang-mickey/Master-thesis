from this import d
import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import os, argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import WeightedRandomSampler

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.dataset.augDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from inference.inference import *
from lib.utils.metrics import *
from lib.utils.CRF import apply_crf
from lib.utils.image_mask_visualize import *


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, iter_warmup=0.0, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.iter_warmup = int(iter_warmup)
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    # Dataset
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,
                                                                      "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
                        help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
                        help="Path to the image dataset folder")
    parser.add_argument("--save_model_path", type=str, default=os.path.join(project_root, "model/model_supervised.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=False, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")

    parser.add_argument("--Dutch", type=bool, default=True, help="use Dutch non-smoke or not")
    parser.add_argument("--Dutch_path", type=str, default=os.path.join(project_root, "lib/dataset/frames/"),
                        help="path to Dutch")

    # Train
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="val batch size")
    parser.add_argument("--augmentation", type=bool, default=True, help="use aug or not")

    parser.add_argument("--num_epochs", type=int, default=20, help="epoch number")
    parser.add_argument("--lr", type=float, default=0.05, help="initial learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    # parser.add_argument("--backbone", type=str, default="Seg", help="choose backone")
    # parser.add_argument("--backbone", type=str, default="deeplabv3plus_Xception", help="choose backone")
    parser.add_argument("--backbone", type=str, default="deeplabv3plus_resnet101", help="choose backone")

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument("--loss_type", type=str, default='BCE',
                        choices=['dice', 'BCE', 'focal_loss'], help="loss type (default: False)")

    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr_patience', default=5, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--iter_size', default=2, type=int, help='when iter_size, opt step forward')
    parser.add_argument('--freeze', default=True, type=bool)
    parser.add_argument("--lr_scheduler", type=str, default='step', choices=['poly', 'step'],
                        help="learning rate scheduler policy")

    return parser.parse_args()


if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()
    print(torch.cuda.is_available())

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    # ---- preprocess ----
    # Get transformations
    image_transform, mask_transform = get_transforms(args.img_size)

    # Split dataset
    train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

    print(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    # Load dataset with transformations
    # dataset = SmokeDataset(json_path, image_folder, transform=image_transform, mask_transform=mask_transform)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load dataset
    train_dataset = SmokeDataset(args.json_path, args.image_folder,
                                 args.smoke5k_path, args.Rise_path,
                                 transform=image_transform,
                                 mask_transform=mask_transform,
                                 image_ids=train_ids,
                                 return_both=args.augmentation,
                                 smoke5k=args.smoke5k, Rise=args.Rise)

    print(f"Total train samples: {len(train_dataset)}")

    random_indices = random.sample(range(len(train_dataset)), 10)

    # Visualize each
    for idx in random_indices:
        show_image_mask(train_dataset, idx)

    val_dataset = SmokeDataset(args.json_path, args.image_folder,
                               args.smoke5k_path, args.Rise_path,
                               transform=image_transform,
                               mask_transform=mask_transform,
                               image_ids=val_ids
                               )

    test_dataset = SmokeDataset(args.json_path, args.image_folder,
                                args.smoke5k_path, args.Rise_path,
                                transform=image_transform,
                                mask_transform=mask_transform,
                                image_ids=test_ids
                                )

    # original_samples = sum(1 for item in train_dataset.image_data if item['source'] == 'coco')
    # smoke5k_samples = sum(1 for item in train_dataset.image_data if item['source'] == 'smoke5k')

    # original_weight = 1.0
    # smoke5k_weight = original_samples / (smoke5k_samples +1e-6)

    # weights = [
    # original_weight if item["source"] == "original" else smoke5k_weight
    # for item in train_dataset.image_data
    # ]

    # sampler = WeightedRandomSampler(
    #     weights=weights,                  # List of weights per sample
    #     num_samples=len(train_dataset), # Total samples per "epoch"
    #     replacement=True                  # re-sampling of minority classes
    # )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True
                              # sampler=sampler
                              #  drop_last=True
                              )
    print(f"Total train batches: {len(train_loader)}")

    val_loader = DataLoader(val_dataset,
                            batch_size=args.val_batch_size,
                            shuffle=False
                            )

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False
                             )

    # ---- Load DeepLabV3+ Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = choose_backbone(args.backbone)
    model.to(device)

    # ---- Define Loss & Optimizer ----
    if args.loss_type == 'dice':
        criterion = dice_loss

    elif args.loss_type == 'BCE':
        criterion = BCEWithLogitsLoss()

    elif args.loss_type == 'focal_loss':
        criterion = dice_loss
        # criterion = FocalLoss()

    optimizer = optim.SGD(
        # params=[
        #     {'params': model.backbone.parameters(), 'lr': args.lr / 10},
        #     {'params': model.classifier.parameters(), 'lr': args.lr}
        # ],
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.lr_scheduler == 'poly':
        scheduler = PolyLR(optimizer, max_iters=len(train_loader) * args.num_epochs, power=0.9)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=args.lr_patience)

    # ---- Training Loop ----
    # max_batches = 1
    best_score = 0.0
    for epoch in range(1, (args.num_epochs + 1)):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0

        for i, (images, masks, _) in enumerate(train_loader):
            # if i >= max_batches:
            #     break  # Stop after two batches

            images = images.to(device)
            masks = masks.to(device)
            # Reset gradients
            optimizer.zero_grad()

            outputs = model(images)
            if args.loss_type == 'dice':
                num_masks = outputs.size(0)
                # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
                loss = criterion(outputs.squeeze(1), masks.squeeze(1), num_masks)  # Remove channel dimension
            else:
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))
            loss.backward()
            # update weights
            optimizer.step()
            if args.lr_scheduler == 'poly':
                scheduler.step()
            # Update loss
            running_loss += loss.item()

            # Calculate metrics
            acc = calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
            iou = calculate_iou(outputs.squeeze(1), masks.squeeze(1))

            if isinstance(acc, torch.Tensor):
                train_accuracy += acc.item()
            else:
                train_accuracy += acc
            if isinstance(iou, torch.Tensor):
                train_iou += iou.item()
            else:
                train_iou += iou

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for i, (images, masks, _) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                num_masks = outputs.size(0)
            # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
            if args.loss_type == 'dice':
                num_masks = outputs.size(0)
                # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
                loss = criterion(outputs.squeeze(1), masks.squeeze(1), num_masks)  # Remove channel dimension

            else:
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))

            val_loss += loss.item()

            # Calculate metrics
            acc = calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
            iou = calculate_iou(outputs.squeeze(1), masks.squeeze(1))

            val_accuracy += acc
            val_iou += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        val_miou = val_iou / len(val_loader)

        print(
            f"Epoch {epoch - 1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Train mIoU: {avg_train_iou:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, Val mIoU: {val_miou:.4f}")

        if args.lr_scheduler == 'step':
            scheduler.step()

    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{os.path.basename(args.save_model_path)}"
    )
    torch.save(model.state_dict(), save_path)
    print("Training complete!")

    # ---- Inference----

    # ---- Load the trained model for testing ----
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # ---- Testing Loop ----
    test_loss = 0.0
    test_accuracy = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_f1 = 0.0

    test_accuracy_crf = 0.0
    test_iou_crf = 0.0
    test_dice_crf = 0.0
    test_f1_crf = 0.0

    with torch.no_grad():
        for i, (images, masks, _) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            num_masks = outputs.size(0)

            if args.loss_type == 'dice':
                num_masks = outputs.size(0)
                # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
                loss = criterion(outputs.squeeze(1), masks.squeeze(1), num_masks)  # Remove channel dimension

            else:
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))
            # Calculate metrics
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
            test_iou += calculate_iou(outputs.squeeze(1), masks.squeeze(1))

            # Additional metrics
            test_dice += calculate_dice(outputs.squeeze(1), masks.squeeze(1))
            test_f1 += calculate_f1(outputs.squeeze(1), masks.squeeze(1))

            # Apply CRF post-processing
            for j in range(images.size(0)):
                # Get original image and convert to numpy
                orig_img = images[j].cpu().numpy().transpose(1, 2, 0)
                # Denormalize the image if needed
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                orig_img = std * orig_img + mean
                orig_img = np.clip(orig_img, 0, 1) * 255.0
                orig_img = orig_img.astype(np.uint8)

                # Get probability map
                prob_map = torch.sigmoid(outputs[j, 0]).cpu().numpy()

                # Apply CRF
                refined_mask = apply_crf(orig_img, prob_map)

                # Convert to tensor and move to device
                refined_mask_tensor = torch.from_numpy(refined_mask).float().to(device)

                # Calculate metrics with CRF
                mask_j = masks[j, 0]
                test_accuracy_crf += calculate_accuracy(refined_mask_tensor, mask_j)
                test_iou_crf += calculate_iou(refined_mask_tensor, mask_j)
                # test_dice_crf += calculate_dice(refined_mask_tensor, mask_j)
                # test_f1_crf += calculate_f1(refined_mask_tensor, mask_j)

    # Calculate averages
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_accuracy / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    avg_test_f1 = test_f1 / len(test_loader)

    # Calculate CRF averages
    avg_test_acc_crf = test_accuracy_crf / len(test_dataset)
    avg_test_iou_crf = test_iou_crf / len(test_dataset)
    # avg_test_dice_crf = test_dice_crf / len(test_dataset)
    # avg_test_f1_crf = test_f1_crf /len(test_dataset)

    print("\nTest Results:")
    print(f"Loss: {avg_test_loss:.4f} | Acc: {avg_test_acc:.4f} | mIoU: {avg_test_iou:.4f}")
    print(f"Dice: {avg_test_dice:.4f} | F1: {avg_test_f1:.4f}")

    print("\nTest Results with CRF:")
    print(f"Acc: {avg_test_acc_crf:.4f} | mIoU: {avg_test_iou_crf:.4f}")
    # print(f"Dice: {avg_test_dice_crf:.4f} | F1: {avg_test_f1_crf:.4f}")
    print("Testing complete!")
