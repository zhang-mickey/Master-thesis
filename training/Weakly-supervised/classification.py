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
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from matplotlib import pyplot as plt

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)
from lib.network.AffinityNet.resnet38_cls import *
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
from inference.inference import *
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
                        default=os.path.join(project_root, "model/model_classification.pth"),
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
    parser.add_argument("--warmup_lr", type=float, default=0.005, help="learning rate")

    parser.add_argument("--num_epochs", type=int, default=10, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    # parser.add_argument("--backbone", type=str, default="resnet101",
    #                     help="choose backone")

    parser.add_argument("--backbone", type=str, default="transformer",
                        help="choose backone")

    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    #                     help="choose backone")

    # parser.add_argument("--CAM_type", type=str, default='TransCAM',
    #                     choices=['grad', 'TransCAM', 'TsCAM'],
    #                     help="CAM type")

    # parser.add_argument("--backbone", type=str, default="conformer",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="resnet38d",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vgg16d",
    #                     help="choose backone")
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

    if args.use_crop:
        train_dataset = aug_CropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size)
        )

        total_size = len(train_dataset)
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size

        # Split dataset
        train_subset, test_subset = random_split(train_dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

        train_dataset_1 = CropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size)
        )

        total_size = len(train_dataset_1)
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size

        # Split dataset
        train_subset_1, test_subset_1 = random_split(train_dataset_1, [train_size, test_size])

        # Create DataLoaders
        train_loader_1 = DataLoader(train_subset_1, batch_size=args.batch_size, shuffle=True)
        test_loader_1 = DataLoader(test_subset_1, batch_size=args.batch_size, shuffle=False)


    #
    #  resize
    else:
        train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

        print(f"Smoke Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

        non_smoke_train, non_smoke_val, non_smoke_test = split_non_smoke_dataset(args.non_smoke_image_folder)

        print(
            f"Non-smoke Dataset split: Train={len(non_smoke_train)}, Val={len(non_smoke_val)}, Test={len(non_smoke_test)}")

        # Load smoke images and extract their masks.
        train_smoke = SmokeDataset(
            args.json_path,
            args.image_folder,
            args.smoke5k_path, args.Rise_path,
            transform=image_transform,
            mask_transform=mask_transform,
            image_ids=train_ids)

        # Load non-smoke images.
        # Select random smoke masks and paste them onto non-smoke images.
        # Ensure realistic blending so that pasted smoke integrates well with the background.
        # Return the modified images and their new mask labels.
        # smoke_aug = SmokeCopyPaste(train_smoke, p=0.7)

        train_dataset = SmokeWeaklyDataset(args.json_path,
                                           args.image_folder,
                                           args.Rise_path,
                                           transform=image_transform,
                                           mask_transform=mask_transform,
                                           image_ids=train_ids,
                                           non_smoke_image_folder=args.non_smoke_image_folder,
                                           non_smoke_files=non_smoke_train,
                                           #    smoke_dataset=train_smoke,
                                           flag=True
                                           , Rise=args.Rise)

        # random_indices = random.sample(range(len(train_dataset)), 10)

        # # Visualize each
        # for idx in random_indices:
        #     show_image_mask_class(train_dataset, idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        original_train_loader = DataLoader(train_smoke, batch_size=args.batch_size, shuffle=True)

        # print(f"Training Dataset loaded: {len(train_dataset)} images in total")
        print(f"Number of batches: {len(train_loader)}")

    # val_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder,
    #                                  transform=image_transform,
    #                                  mask_transform=mask_transform,
    #                                  image_ids=val_ids,
    #                                  non_smoke_image_folder=args.non_smoke_image_folder,
    #                                  non_smoke_files=non_smoke_val)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # print(f"Validation Dataset loaded: {len(val_dataset)} images in total")

    # test_dataset = SmokeWeaklyDataset(args.json_path, args.image_folder,
    #                                   transform=image_transform,
    #                                   mask_transform=mask_transform,
    #                                   image_ids=test_ids,
    #                                   non_smoke_image_folder=args.non_smoke_image_folder,
    #                                   non_smoke_files=non_smoke_test)

    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # print(f"Test Dataset loaded: {len(test_dataset)} images in total")

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

    avg_meter = AverageMeter('loss', 'accuracy', 'loss_entropy', 'bg_loss')
    # max_batches = 2

    if args.backbone == 'vgg16:':
        for epoch in range(1, (args.num_epochs + 1)):
            avg_meter.pop()
            data_load_start = time.time()
            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                # if batch_idx >= max_batches:
                #     break  # Stop after two batches
                images, labels = images.to(device), labels.float().to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                outputs = outputs.squeeze(1)
                acc = calculate_accuracy(outputs, labels)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Update AverageMeter with loss and accuracy
                avg_meter.add({'loss': loss.item(), 'accuracy': acc})
                # if batch_idx % 10 == 0:
                #     print(f"Epoch [{epoch}/{args.num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

            scheduler.step()
            avg_loss, avg_acc = avg_meter.get('loss', 'accuracy')
            print(f"Epoch [{epoch}/{args.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        save_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
        )
        torch.save(model.state_dict(), save_path)
        print("Training complete! Model saved.")

        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.cuda()

        # infer phase
        print("Starting test phase...")
        test_loss = 0.0
        test_accuracy = 0.0
        test_predictions = []
        test_ground_truth = []
        iou_sum = 0.0
        total_samples = 0
        for batch_idx, (images, labels, image_ids, masks) in enumerate(train_loader):
            # (batchsize,channel,height,width)
            images = images.to(device)  # [B, num_classes]
            masks = masks.to(device)
            orig_img_size = images.shape[2:]
            # with torch.no_grad():
            if True:
                for i, (img, label, img_id, mask) in enumerate(zip(images, labels, image_ids, masks)):
                    image = img
                    img = img.unsqueeze(0)

                    grad_cam = compute_grad_cam(model, img)
                    grad_cam = F.interpolate(grad_cam, size=orig_img_size,
                                             mode='bilinear', align_corners=False)[0]  # [C, H, W]
                    # [1, H, W]
                    grad_cam = grad_cam[0]  # [H, W]
                    grad_cam = grad_cam.cpu().numpy()

                    with torch.no_grad():
                        outputs = model(img)
                        cams = model.forward_cam(img)
                        cam = F.interpolate(cams, size=orig_img_size,
                                            mode='bilinear', align_corners=False)[0]  # [C, H, W]
                        cam = cam[0]  # [H, W]
                        cam = cam.cpu().numpy()

                        mean = np.array([0.485, 0.456, 0.406])  # Adjust based on your dataset
                        std = np.array([0.229, 0.224, 0.225])  # Adjust based on your dataset

                        img_np = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                        img_np = std * img_np + mean  # Denormalize
                        img_np = np.clip(img_np, 0, 1)

                        pseudo_label = (cam > args.threshold).astype(np.float32)

                        # cam_list.append(cam)
                        gt_mask = mask.squeeze().cpu().numpy()
                        gt_mask = (gt_mask > 0.5).astype(np.float32)  # Ensure binary mask

                        intersection = np.logical_and(gt_mask, pseudo_label).sum()
                        union = np.logical_or(gt_mask, pseudo_label).sum()
                        iou = intersection / (union + 1e-8)  # Avoid division by zero
                        iou_sum += iou
                        total_samples += 1
                        fig, ax = plt.subplots(1, 4, figsize=(25, 5))

                        ax[0].imshow(img_np)
                        ax[0].set_title('Original Image')
                        ax[0].axis('off')

                        # CAM visualization
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        ax[1].imshow(cam, cmap='jet')
                        ax[1].set_title('Class Activation Map')
                        ax[1].axis('off')

                        # # Pseudo mask
                        # ax[2].imshow(pseudo_label, cmap='gray')
                        # ax[2].set_title(f'Pseudo Mask (IoU: {iou:.2f})')
                        # ax[2].axis('off')
                        ax[2].imshow(grad_cam, cmap='jet')
                        ax[2].set_title('Grad_CAM')
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

                        pseudo_label_dir = os.path.join(args.save_pseudo_labels_path, args.backbone)
                        os.makedirs(pseudo_label_dir, exist_ok=True)

                        cv2.imwrite(
                            os.path.join(pseudo_label_dir, f"pseudo_label_{img_id}.png"),
                            (pseudo_label * 255).astype(np.uint8)
                        )

                        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                        cam_dir = os.path.join(args.save_cam_path, args.backbone)
                        os.makedirs(cam_dir, exist_ok=True)

                        cv2.imwrite(
                            os.path.join(cam_dir, f"cam_vis_{img_id}.png"),
                            heatmap
                        )

                # print(f"Batch {batch_idx+1}/{len()} - Current Mean IoU: {iou_sum/total_samples:.4f}")

        # Final statistics
        print(f"\nFinal Mean IoU: {iou_sum / total_samples:.4f} (over {total_samples} samples)")




    elif args.backbone == 'transformer' or args.backbone == 'mix_transformer' or args.backbone == 'resnet101' or args.backbone == 'resnet38d':

        cls_loss_history = []
        bg_loss_history = []
        train_accuracies = []
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()
            data_load_start = time.time()

            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                # print(outputs.shape)
                if isinstance(outputs, tuple):
                    attns = outputs[1]
                    last_attn = attns[-1]
                    dcp_map = []
                    for img in images.cpu().numpy():
                        img = img.transpose(1, 2, 0)
                        dcp_map.append(get_dark_channel(img))
                    dcp_map = np.stack(dcp_map, axis=0)
                    dcp_map = torch.tensor(dcp_map).to(device)

                    # dcp_loss=dcp_guidance_loss(last_attn,dcp_map)
                    bg_loss = background_suppression_loss(last_attn, dcp_map)
                    loss_entropy = attention_entropy_loss(last_attn)
                    outputs = outputs[0]

                # outputs=outputs[-1]
                outputs = outputs.squeeze(1)

                acc = calculate_accuracy(outputs, labels)
                # if epoch<=10:
                #     #Use DCP only as a soft regularizer (early stages only)
                #     loss = criterion(outputs, labels)
                # # elif epoch<=8:
                # #     # loss = criterion(outputs, labels)+0.1*loss_entropy+1*bg_loss
                # #     loss = criterion(outputs, labels)+0.1*loss_entropy+1*bg_loss
                # else:
                #     loss = criterion(outputs, labels)+1*bg_loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_meter.add({'loss': loss.item(), 'accuracy': acc,
                               'loss_entropy': loss_entropy.item(),
                               'bg_loss': bg_loss.item()})

            data_load_time = time.time() - data_load_start

            print(f"Data loading time: {data_load_time:.2f} seconds")

            avg_loss, avg_acc, avg_loss_entropy, avg_bg_loss = avg_meter.get('loss', 'accuracy', 'loss_entropy',
                                                                             'bg_loss')
            print(
                f"Epoch [{epoch}/{args.num_epochs}], Avg cls Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%,Avg loss_entropy: {avg_loss_entropy:.4f},Avg bg_loss: {avg_bg_loss:.4f}")

            cls_loss_history.append(avg_loss)
            bg_loss_history.append(avg_bg_loss)
            train_accuracies.append(avg_acc)
            scheduler.step()

        save_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
        )
        save_loss_path = os.path.join(
            os.path.dirname(args.save_visualization_path),
            f"loss_{args.backbone}_{args.num_epochs}_{os.path.basename(args.save_visualization_path)}"
        )

        plot_loss_accuracy(cls_loss_history, bg_loss_history, train_accuracies, save_loss_path, args.lr)

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
            for batch_idx, (images, labels, _, mask) in enumerate(test_loader_1):
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

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Test Batch [{batch_idx + 1}/{len(test_loader_1)}], ClS Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        # Calculate average metrics
        avg_test_loss = test_loss / len(test_loader_1)
        avg_test_accuracy = test_accuracy / len(test_loader_1)

        # precision = precision_score(test_ground_truth, test_predictions, zero_division=0)
        # recall = recall_score(test_ground_truth, test_predictions, zero_division=0)
        f1 = f1_score(test_ground_truth, test_predictions, zero_division=0)
        # conf_matrix = confusion_matrix(test_ground_truth, test_predictions)

        print("\n===== Test Results =====")
        print(f"Average CLS Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {avg_test_accuracy:.2f}%")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        # print(f"Confusion Matrix:\n{conf_matrix}")

        # Determine target layers for CAM
        # if args.backbone == "resnet101":
        #     target_layers = [model.layer4[-1]]  # Last layer of layer4
        # elif args.backbone == "transformer":
        #     target_layers = [model.blocks[-1].norm1]  # Last transformer block
        # elif args.backbone == "mix_transformer":
        #     target_layers = [model.blocks[-1].norm1]  # Last transformer block
        # elif args.backbone == "resnet38d":
        #     target_layers = [model.dropout7]

        # save_cam_path = os.path.join(
        #     os.path.dirname(args.save_cam_path),
        #     f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}"
        # )

        # if model is None:
        #     print("ERROR: Model is None before CAM generation!")
        #     # Either exit or fix the model
        # else:
        #     print(f"Model type: {type(model)}")
        #     print(f"Target layers: {target_layers}")

        # generate_cam_for_dataset(
        #     dataloader=train_loader,
        #     model=model,
        #     target_layers=target_layers,
        #     save_dir=save_cam_path,
        # )

        # Generate pseudo-labels
        # generate_pseudo_labels(
        #     dataloader=original_train_loader,
        #     model=model,
        #     target_layers=target_layers,
        #     save_dir=args.save_pseudo_labels_path,
        #     threshold=args.threshold
        # )

    elif args.CAM_type == 'TransCAM':
        for epoch in range(1, (args.num_epochs + 1)):

            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)

                logits_conv, logit_trans, cams = model(args.CAM_type, images)
                # Combine both logits for final prediction
                combined_logits = logits_conv + logit_trans

                loss = criterion(combined_logits.squeeze(1), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = calculate_accuracy(combined_logits, labels)
                avg_meter.add({'loss': loss.item(), 'accuracy': acc})

                # if (batch_idx + 1) % 10 == 0:
                #     print(f"Epoch [{epoch}/{args.num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}], "
                #           f"Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

            avg_loss, avg_acc = avg_meter.get('loss', 'accuracy')
            print(f"Epoch [{epoch}/{args.num_epochs}], Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%")
        save_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
        )
        torch.save(model.state_dict(), save_path)
        print("Training complete! Model saved.")

        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.cuda()

        iou_sum = 0.0
        total_samples = 0
        for batch_idx, (images, labels, image_ids, masks) in enumerate(original_train_loader):
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

                    mean = np.array([0.485, 0.456, 0.406])  # Adjust based on your dataset
                    std = np.array([0.229, 0.224, 0.225])  # Adjust based on your dataset

                    img_np = img.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                    img_np = std * img_np + mean  # Denormalize
                    img_np = np.clip(img_np, 0, 1)

                    pseudo_label = (cam > args.threshold).astype(np.float32)
                    # cam_list.append(cam)
                    gt_mask = mask.squeeze().cpu().numpy()
                    gt_mask = (gt_mask > 0.5).astype(np.float32)  # Ensure binary mask

                    intersection = np.logical_and(gt_mask, pseudo_label).sum()
                    union = np.logical_or(gt_mask, pseudo_label).sum()
                    iou = intersection / (union + 1e-8)  # Avoid division by zero
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

                    cv2.imwrite(
                        os.path.join(args.save_pseudo_labels_path, f"pseudo_label_{img_id}.png"),
                        (pseudo_label * 255).astype(np.uint8)
                    )

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

                    cv2.imwrite(
                        os.path.join(args.save_cam_path, f"cam_vis_{img_id}.png"),
                        heatmap
                    )

            # print(f"Batch {batch_idx+1}/{len(dataloader)} - Current Mean IoU: {iou_sum/total_samples:.4f}")

    # # Final statistics
    # print(f"\nFinal Mean IoU: {iou_sum/total_samples:.4f} (over {total_samples} samples)")

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
    #     np.save(os.path.join(args.save_cam_path, f"cams_{batch_idx}.npy"), cam_dict)







