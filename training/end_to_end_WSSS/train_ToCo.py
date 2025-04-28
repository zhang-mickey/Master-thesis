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
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import sklearn.metrics as metrics

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)
from lib.dataset.cropDataset import *
from lib.dataset.SmokeDataset import *
from lib.dataset.augDataset import *
from lib.dataset.localcropDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from inference.inference import *
from lib.utils.metrics import *
from lib.utils.CRF import *
from lib.utils.PAR import *
from lib.utils.image_mask_visualize import *
from lib.network.ToCo_model import *


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
    parser.add_argument("--json_path", type=str,
                        default=os.path.join(project_root,
                                             "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
                        help="Path to COCO annotations JSON file")

    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
                        help="Path to the image dataset folder")

    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_supervised.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--supervised_crf_mask_path", type=str,
                        default=os.path.join(project_root, "result/supervised/crf_mask"),
                        help="Path to save the CRF cam")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=True, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")

    parser.add_argument("--Dutch", type=bool, default=True, help="use Dutch non-smoke or not")
    parser.add_argument("--Dutch_path", type=str, default=os.path.join(project_root, "lib/dataset/frames/"),
                        help="path to Dutch")

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

    # Train
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="val batch size")
    parser.add_argument("--augmentation", type=bool, default=True, help="use aug or not")

    parser.add_argument("--num_epochs", type=int, default=15, help="epoch number")
    parser.add_argument("--lr", type=float, default=0.05, help="initial learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")

    parser.add_argument("--backbone", type=str, default="ToCo",
                        help="choose backone")
    parser.add_argument("--ignore_index", default=255, type=int, help="random index")
    parser.add_argument('--high_threshold', default=0.65, type=float,
                        help='high threshold')
    parser.add_argument('--low_threshold', default=0.25, type=float,
                        help='low threshold')
    parser.add_argument('--background_threshold', default=0.45, type=float,
                        help='background threshold')

    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5),
                        help="multi_scales for cam")

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
    print(vars(args))

    print(torch.cuda.is_available())

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- preprocess ----
    # Get transformations
    image_transform, mask_transform = get_transforms(args.img_size)

    # Split dataset
    if args.use_crop:
        train_dataset = LocalCropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            transform=image_transform,
            mask_transform=mask_transform,
            local_crop_size=args.local_crop_size,
        )
        total_size = len(train_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        train_subset, val_subset, test_subset = random_split(train_dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

        print(f"Train size: {len(train_subset)} |Test size: {len(val_subset)} |Test size: {len(test_subset)}")
    else:
        train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

        print(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

        # Load dataset
        train_dataset = SmokeDataset(args.json_path, args.image_folder,
                                     args.smoke5k_path, args.Rise_path,
                                     transform=image_transform,
                                     mask_transform=mask_transform,
                                     image_ids=train_ids,
                                     return_both=args.augmentation,
                                     smoke5k=args.smoke5k, Rise=args.Rise)

        print(f"Total train samples: {len(train_dataset)}")

        # random_indices = random.sample(range(len(train_dataset)), 10)

        # # Visualize each
        # for idx in random_indices:
        #     show_image_mask(train_dataset, idx)

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

        original_samples = sum(1 for item in train_dataset.image_data if item['source'] == 'coco')
        smoke5k_samples = sum(1 for item in train_dataset.image_data if item['source'] == 'smoke5k')

        original_weight = 1.0
        smoke5k_weight = original_samples / (smoke5k_samples + 1e-6)

        weights = [
            original_weight if item["source"] == "original" else smoke5k_weight
            for item in train_dataset.image_data
        ]

        sampler = WeightedRandomSampler(
            weights=weights,  # List of weights per sample
            num_samples=len(train_dataset),  # Total samples per "epoch"
            replacement=True  # re-sampling of minority classes
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  # shuffle=True
                                  sampler=sampler
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

    # ---- Model ----

    model = choose_backbone(args.backbone)
    model.to(device)

    # ---- Define Loss & Optimizer ----
    avg_meter = AverageMeter('cls_loss', 'ptc_loss', 'ctc_loss', 'cls_loss_aux', 'seg_loss', 'cls_score', 'total_loss')

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    ncrops = 10

    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=1.0).cuda()

    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

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
    best_score = 0.0
    for epoch in range(1, (args.num_epochs + 1)):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0

        for i, (images, cls_labels, image_ids, masks, crops) in enumerate(train_loader):
            images = images.to(device)
            images_denorm = denormalize_img2(images)
            cls_labels = cls_labels.to(device)

            masks = masks.to(device)
            # Reset gradients
            optimizer.zero_grad()

            cams, cams_aux = multi_scale_cam2(model, inputs=images, scales=args.cam_scales)

            roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_labels,
                                        hig_thre=args.high_threshold,
                                        low_thre=args.low_threshold)

            local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask,
                                                   crop_num=ncrops - 2, crop_size=96)
            roi_crops = crops[:2] + local_crops
            roi_crops = [crop.to(device) for crop in roi_crops]
            n_iter = (epoch - 1) * len(train_loader) + i

            cls, segs, fmap, cls_aux, out_t, out_s = model(images, crops=roi_crops, n_iter=n_iter)

            # print("cls",cls.shape)
            #             print("cls_labels",cls_labels.shape)
            #             print("cls_aux",cls_aux.shape)
            #             cls torch.Size([8, 2])
            #             cls_labels torch.Size([8])
            #              cls_aux torch.Size([8, 2])
            if cls_labels.ndim == 1:
                cls_labels = F.one_hot(cls_labels, num_classes=2).float()
            cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

            cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_labels)

            ctc_loss = CTC_loss(out_s, out_t, flags)

            valid_cam, _ = cam_to_label(
                cams.detach(),
                cls_label=cls_labels,
                # img_box=None,
                ignore_mid=True,
                bkg_thre=args.background_threshold,
                high_thre=args.high_threshold,
                low_thre=args.low_threshold,
                ignore_index=args.ignore_index)

            valid_cam_aux, _ = cam_to_label(
                cams_aux.detach(),
                cls_label=cls_labels,
                # img_box=None,
                ignore_mid=True,
                bkg_thre=args.background_threshold,
                high_thre=args.high_threshold,
                low_thre=args.low_threshold,
                ignore_index=args.ignore_index)

            if epoch <= 7:
                refined_pseudo_label = refine_cams_with_bkg_v2(par, images_denorm, cams=valid_cam_aux,
                                                               cls_labels=cls_labels,
                                                               high_thre=args.high_threshold,
                                                               low_thre=args.low_threshold,
                                                               ignore_index=args.ignore_index,
                                                               # img_box=img_box,
                                                               )
            else:
                refined_pseudo_label = refine_cams_with_bkg_v2(par, images_denorm, cams=valid_cam,
                                                               cls_labels=cls_labels,
                                                               high_thre=args.high_threshold,
                                                               low_thre=args.low_threshold,
                                                               ignore_index=args.ignore_index,
                                                               # img_box=img_box,
                                                               )
            # print("refined_pseudo_label.shape",refined_pseudo_label.shape)
            print(f"Label min: {refined_pseudo_label.min().item()}, max: {refined_pseudo_label.max().item()}")
            print(f"Number of classes in segs: {segs.size(1)}")
            print(f"segs shape: {segs.shape}")

            segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

            # reg_loss = get_energy_loss(
            #     img=images,
            #     logit=segs,
            #     label=refined_pseudo_label,
            #     # img_box=None,
            #     loss_layer=loss_layer)

            resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)

            _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_labels,
                                               # img_box=None,
                                               ignore_mid=True,
                                               bkg_thre=args.background_threshold,
                                               high_thre=args.high_threshold,
                                               low_thre=args.low_threshold,
                                               ignore_index=args.ignore_index)
            print("pseudo_label_aux", pseudo_label_aux.shape)

            aff_mask = label_to_aff_mask(pseudo_label_aux)

            ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
            # ptc_loss = get_ptc_loss(fmap, low_fmap)
            # Early training phase - Using only classification losses")
            if epoch <= 5:
                loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.0 * seg_loss
                # loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.0 * seg_loss + 0.0 * reg_loss
            # Middle training phase - Adding segmentation and regularization losses"
            elif epoch <= 9:
                loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.1 * seg_loss
                # loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.1 * seg_loss + args.w_reg * reg_loss
            # Final training phase - Using all losses
            else:
                # loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.5 * ctc_loss + 0.1 * seg_loss + args.w_reg * reg_loss
                loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.5 * ctc_loss + 0.1 * seg_loss

            cls_pred = (cls > 0).type(torch.int16)
            cls_score = metrics.f1_score(cls_labels.cpu().numpy().flatten(), cls_pred.cpu().numpy().flatten())

            avg_meter.add({
                'cls_loss': cls_loss.item(),
                'ptc_loss': ptc_loss.item(),
                'ctc_loss': ctc_loss.item(),
                'cls_loss_aux': cls_loss_aux.item(),
                'seg_loss': seg_loss.item(),
                'cls_score': cls_score.item(),
                'total_loss': loss.item(),
            })

            loss.backward()
            # update weights
            optimizer.step()
            if args.lr_scheduler == 'poly':
                scheduler.step()

        #     iou = calculate_iou(outputs.squeeze(1), masks.squeeze(1))

        #     if isinstance(iou, torch.Tensor):
        #         train_iou += iou.item()
        #     else:
        #         train_iou += iou
        # avg_train_iou = train_iou / len(train_loader)

        # Validation Phase

        # model.eval()
        # # val_loss = 0.0
        # # val_iou = 0.0
        # with torch.no_grad():
        #     for i,(images, cls_labels, _, masks,crops) in enumerate(val_loader):
        #         images, masks = images.to(device), masks.to(device)
        #         cls_labels = cls_labels.to(device)
        #         # num_masks=outputs.size(0)
        #     # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
        #     if args.loss_type == 'dice':
        #         num_masks=outputs.size(0)
        #     # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
        #         loss = criterion(outputs.squeeze(1), masks.squeeze(1),num_masks)  # Remove channel dimension

        #     else:
        #         loss =criterion(outputs.squeeze(1), masks.squeeze(1))

        #     val_loss += loss.item()

        #     # Calculate metrics
        #     acc = calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
        #     iou = calculate_iou(outputs.squeeze(1), masks.squeeze(1))

        #     val_accuracy += acc
        #     val_iou += iou

        # avg_val_loss = val_loss / len(val_loader)
        # avg_val_accuracy = val_accuracy / len(val_loader)
        # val_miou = val_iou / len(val_loader)
        # print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, Val mIoU: {val_miou:.4f}")
        print(f"Epoch: {epoch}/{args.num_epochs} "
              f"Loss: {avg_meter.get('total_loss'):.4f}, "
              f"Cls Score: {avg_meter.get('cls_score'):.4f}")
        if args.lr_scheduler == 'step':
            scheduler.step()

    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )
    torch.save(model.state_dict(), save_path)
    print("Training complete!")

    # ---- Inference----

    # ---- Load the trained model for testing ----
    # model.load_state_dict(torch.load(save_path))
    # model.eval()

    #     # ---- Testing Loop ----
    # test_loss = 0.0
    # test_accuracy = 0.0
    # test_iou = 0.0
    # test_dice = 0.0
    # test_f1 = 0.0

    # with torch.no_grad():
    #     for i, (images, labels, _, masks) in enumerate(test_loader):
    #         images = images.to(device)
    #         masks = masks.to(device)

    #         #foreground_prob
    #         outputs = model(images)

    #         if args.loss_type == 'dice':
    #             num_masks=outputs.size(0)
    #         # print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")
    #             loss = criterion(outputs.squeeze(1), masks.squeeze(1),num_masks)  # Remove channel dimension

    #         else:
    #             loss =criterion(outputs.squeeze(1), masks.squeeze(1))
    #         # Calculate metrics
    #         test_loss += loss.item()
    #         test_accuracy += calculate_accuracy(outputs.squeeze(1), masks.squeeze(1))
    #         test_iou += calculate_iou(outputs.squeeze(1), masks.squeeze(1))

    #         # Additional metrics
    #         test_dice += calculate_dice(outputs.squeeze(1), masks.squeeze(1))
    #         test_f1 += calculate_f1(outputs.squeeze(1), masks.squeeze(1))

    #         # Apply CRF post-processing
    #         batch_size = images.size(0)
    #         for j in range(batch_size):
    #             #(3,512,512)
    #             image_j = img_denorm(images[j].cpu().numpy()).astype(np.uint8)
    #             mask_j = masks[j].cpu().numpy()

    #             # outputs_j = outputs[j].cpu().numpy()
    #             #per-pixel class probability map (after softmax)
    #             outputs_j = torch.softmax(outputs[j], dim=0).cpu().numpy()
    #             # Handle binary case background score

    #             if outputs_j.shape[0] == 1:
    #                 fg = outputs_j[0]
    #                 bg = 1 - fg
    #                 outputs_j = np.stack([bg, fg], axis=0)

    #             n_classes = outputs_j.shape[0]

    #             crf_outputs=dense_crf(outputs_j,image_j,n_classes=n_classes,n_iter=10)
    #              # Convert CRF output to tensor and move to device
    #             # print("CRF output shape:", crf_outputs.shape) #CRF output shape: (2, 512, 512)

    #             crf_mask = np.argmax(crf_outputs, axis=0)  # (H, W)
    #             # foreground_mask = crf_outputs[1]
    #             # crf_mask= (foreground_mask > 0.5).astype(np.uint8)

    #             # print("Unique values in CRF mask:", np.unique(crf_mask))

    #             refined_mask_tensor = torch.from_numpy(crf_mask).to(device)

    #             mask_j = masks[j].squeeze().long()

    #             refined_mask_tensor = refined_mask_tensor.unsqueeze(0)  # shape [1, H, W]
    #             mask_j = mask_j.unsqueeze(0)  # ensure same shape

    #             save_dir = os.path.join(args.supervised_crf_mask_path, args.backbone)
    #             os.makedirs(save_dir, exist_ok=True)
    #             save_path = os.path.join(save_dir, f"supervised_image_{i}_mask_{j}.png")
    #             fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    #             image_j = image_j.transpose(1, 2, 0)

    #             ax[0].imshow(image_j)
    #             ax[0].set_title('Original Image')
    #             ax[0].axis('off')

    #             ax[1].imshow(crf_mask, cmap='gray')
    #             ax[1].set_title('crf_mask')
    #             ax[1].axis('off')

    #             ax[2].imshow(mask_j.cpu().squeeze().numpy(), cmap='gray')
    #             ax[2].set_title('Ground Truth Mask')
    #             ax[2].axis('off')
    #             prob_map=outputs_j[1]

    #             ax[3].imshow(prob_map.squeeze(), cmap='viridis')
    #             ax[3].set_title('Output Foreground Prob')
    #             ax[3].axis('off')

    #             plt.tight_layout()
    #             plt.savefig(os.path.join(save_dir, f"supervised_image_{i}_mask_{j}.png"), bbox_inches='tight')
    #             plt.close()

    # # Calculate averages
    # avg_test_loss = test_loss / len(test_loader)
    # avg_test_acc = test_accuracy / len(test_loader)
    # avg_test_iou = test_iou / len(test_loader)
    # avg_test_dice = test_dice / len(test_loader)
    # avg_test_f1 = test_f1 / len(test_loader)

    # print("\nTest Results:")
    # print(f"Loss: {avg_test_loss:.4f} | mIoU: {avg_test_iou:.4f}")
    # print(f"Dice: {avg_test_dice:.4f} | F1: {avg_test_f1:.4f}")

