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
from torch.utils.data import random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from matplotlib import pyplot as plt

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)
from lib.network.AffinityNet_resnet.resnet38_cls import *
from lib.dataset.SmokeDataset import *
from lib.dataset.WeaklyDataset import *
from lib.dataset.cropDataset import *
from lib.dataset.aug_cropDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.utils.dark_channel_prior import *
from lib.loss.loss import *
from post_processing.single_image_infer import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.utils.cam import *
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
                        default=os.path.join(project_root, "model/model_classification_raw.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/pseudo_labels"),
                        help="Path to save the pseudo labels")

    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root, "result/cam"),
                        help="Path to save the cam")

    parser.add_argument("--save_visualization_path", type=str,
                        default=os.path.join(project_root, "result/visualization"),
                        help="Path to save the cam")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=False, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")

    parser.add_argument("--Dutch", type=bool, default=True, help="use Dutch non-smoke or not")
    parser.add_argument("--Dutch_negative_path", type=str,
                        default=os.path.join(project_root, "frames/manual_negative/"), help="path to Dutch")
    parser.add_argument("--Dutch_positive_path", type=str,
                        default=os.path.join(project_root, "frames/manual_positive/"), help="path to Dutch")

    parser.add_argument("--crop_smoke_image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images/"),
                        help="Path to the cropped smoke image dataset folder")

    parser.add_argument("--crop_mask_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks/"),
                        help="Path to the cropped image dataset mask folder")

    parser.add_argument("--crop_non_smoke_folder", type=str,
                        default=os.path.join(project_root,
                                             "smoke-segmentation.v5i.coco-segmentation/non_smoke_images/"),
                        help="Path to the cropped image dataset mask folder")

    parser.add_argument("--use_crop", type=bool, default=True, help="use cropped image or not")

    # train
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup_lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--num_epochs", type=int, default=2, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--CAM_type", type=str, default='TransCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    # parser.add_argument("--backbone", type=str, default="resnet101",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vit_b",
    #                     help="choose backone")
    parser.add_argument("--backbone", type=str, default="conformer",
                        help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vit_s",
    #                     help="choose backone")
    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    #                     help="choose backone")

    # parser.add_argument("--CAM_type", type=str, default='TransCAM',
    #                     choices=['grad', 'TransCAM', 'TsCAM'],
    #                     help="CAM type")

    # parser.add_argument("--backbone", type=str, default="resnet38d",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vgg16d",
    #                     help="choose backone")
    parser.add_argument('--manual_seed', default=42, type=int, help='Manually set random seed')

    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')
    parser.add_argument('--ratio', default=0.1, type=float, help='ratio for ood')
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

    if args.use_crop:
        train_dataset = CropDataset(
            # image_dir=args.crop_smoke_image_folder,
            # mask_dir=args.crop_mask_folder,
            # non_smoke_dir=args.crop_non_smoke_folder,
            ijmond_positive_dir=args.Dutch_positive_path,
            ijmond_negative_dir=args.Dutch_negative_path,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size),
            backbone=args.backbone
        )

        total_size = len(train_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_size = total_size

        train_indices = indices[:train_size]

        train_subset = Subset(train_dataset, train_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

        train_dataset_1 = CropDataset(
            image_dir=args.crop_smoke_image_folder,
            mask_dir=args.crop_mask_folder,
            # non_smoke_dir=args.crop_non_smoke_folder,
            # ijmond_positive_dir=args.Dutch_positive_path,
            # ijmond_negative_dir=args.Dutch_negative_path,
            test=True,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size),
            backbone=args.backbone
        )

        total_size_1 = len(train_dataset_1)
        indices_1 = list(range(total_size_1))
        random.shuffle(indices_1)
        test_size_1 = total_size_1
        test_indices_1 = indices_1[:test_size_1]
        test_subset_1 = Subset(train_dataset_1, test_indices_1)

        test_loader = DataLoader(test_subset_1, batch_size=args.batch_size, shuffle=False)

        print(f"Training Dataset loaded: {len(train_dataset)} images in total")
        print(f"Test Dataset loaded: {len(train_dataset_1)} images in total")
        print(f"Number of batches: {len(train_loader)}")
        print(f"Number of batches: {len(test_loader)}")

    model = choose_backbone(args.backbone)
    model = model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # warmup_scheduler=LinearLR(optimizer, start_factor=args.warmup_lr/args.lr, total_iters=5)
    # main_scheduler=CosineAnnealingLR(optimizer, T_max=args.num_epochs-5)

    # scheduler = SequentialLR(
    # optimizer,
    # schedulers=[warmup_scheduler, main_scheduler],
    # milestones=[5]  # Switch to main_scheduler after epoch 5
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter = AverageMeter('loss', 'cls_loss', 'f1', 'accuracy', 'loss_entropy', 'bg_loss', 'consistency_loss')

    if args.CAM_type == 'TransCAM':
        # for epoch in range(1, (args.num_epochs + 1)):

        #     for batch_idx, (images, labels, _, mask) in enumerate(train_loader):

        #         images, labels = images.to(device), labels.float().to(device)

        #         logits_conv, logit_trans, cams = model(args.CAM_type,images)
        #         # Combine both logits for final prediction
        #         combined_logits = logits_conv + logit_trans
        #         logits=combined_logits.squeeze(1)
        #         loss = criterion(logits, labels)

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         acc = calculate_accuracy(logits, labels)
        #         f1=calculate_f1(logits, labels)
        #         avg_meter.add({'loss': loss.item(),'f1':f1, 'accuracy': acc})

        #     scheduler.step()
        #     avg_loss, avg_f1,avg_acc = avg_meter.get('loss', 'f1','accuracy')
        #     print(f"Epoch [{epoch}/{args.num_epochs}], Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%")
        save_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
        )
        # torch.save(model.state_dict(), save_path)
        # print("Training complete! Model saved.")

        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.cuda()

        iou_sum = 0.0
        total_samples = 0
        for batch_idx, (images, labels, image_ids, masks) in enumerate(test_loader):
            # (batchsize,channel,height,width)
            images = images.to(device)  # [B, num_classes]

            masks = masks.to(device)
            # cam_list=[]
            orig_img_size = images.shape[2:]

            with torch.no_grad():
                for i, (img, label, img_id, mask) in enumerate(zip(images, labels, image_ids, masks)):
                    image = img.unsqueeze(0)  # [1, C, H, W]

                    # cam: [1, num_classes, H', W']
                    logits_conv, logit_trans, cam = model(args.CAM_type, image)

                    # Remove background channel if your model outputs it
                    cam = F.interpolate(cam, size=orig_img_size,
                                        mode='bilinear', align_corners=False)[0]  # [C, H, W]

                    cam = cam[0]  # [H, W]
                    cam = cam.cpu().numpy()

                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                    # print("cam_max:",np.max(cam))

                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])

                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)

                    pseudo_label = (cam > args.threshold).astype(np.float32)

                    gt_mask = mask.squeeze().cpu().numpy()
                    gt_mask = (gt_mask > 0.5).astype(np.float32)

                    intersection = np.logical_and(gt_mask, pseudo_label).sum()
                    union = np.logical_or(gt_mask, pseudo_label).sum()
                    iou = intersection / (union + 1e-8)
                    iou_sum += iou
                    total_samples += 1
                    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

                    ax[0].imshow(img_np)
                    ax[0].set_title('Original Image')
                    ax[0].axis('off')

                    # CAM visualization
                    ax[1].imshow(cam, cmap='jet')
                    ax[1].set_title('Class Activation Map')
                    ax[1].axis('off')

                    # Pseudo mask
                    ax[2].imshow(pseudo_label, cmap='gray')
                    ax[2].set_title(f'Pseudo Mask (IoU: {iou:.2f})')
                    ax[2].axis('off')

                    # Ground truth mask
                    ax[3].imshow(gt_mask, cmap='gray')
                    ax[3].set_title('Ground Truth Mask')
                    ax[3].axis('off')

                    save_dir = os.path.join(args.save_visualization_path, args.backbone)
                    os.makedirs(save_dir, exist_ok=True)

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'visualization_{img_id}.png'), bbox_inches='tight')
                    plt.close()

                    # cv2.imwrite(
                    #     os.path.join(args.save_pseudo_labels_path, f"pseudo_label_{img_id}.png"),
                    #     (pseudo_label * 255).astype(np.uint8)
                    # )

                    # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

                    # cv2.imwrite(
                    #     os.path.join(args.save_cam_path, f"cam_vis_{img_id}.png"),
                    #     heatmap
                    # )

            # print(f"Batch {batch_idx+1}/{len(dataloader)} - Current Mean IoU: {iou_sum/total_samples:.4f}")

    # Final statistics
    print(f"\nFinal Mean IoU: {iou_sum / total_samples:.4f} (over {total_samples} samples)")

    # multi-scale
    # sum_cam=np.sum(cam_list,axis=0)
    # sum_cam[sum_cam<0]=0
    # cam_max=np.max(sum_cam,(1,2),keepdims=True)
    # cam_min=np.min(sum_cam,(1,2),keepdims=True)
    # sum_cam[sum_cam<cam_min+1e-5]=0
    # norm_cam=(sum_cam-cam_min)/(cam_max-cam_min+1e-5)

    # cam_dict={}
    # for i in range(1):
    #     if label[i]>1e-5:
    #         cam_dict[i]=norm_cam[i]

    # if args.save_cam_path is not None:
    #     np.save(os.path.join(args.save_cam_path, f"cams_{batch_idx}.npy"), cam_dict


