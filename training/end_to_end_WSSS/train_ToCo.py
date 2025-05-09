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

    parser.add_argument("--num_epochs", type=int, default=20, help="epoch number")
    parser.add_argument("--lr", type=float, default=5e-3, help="initial learning rate")
    # parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    # parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    # parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate") converge but misaligned
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
    parser.add_argument("--crop_size", default=512, type=int, help="crop_size for global view")

    parser.add_argument('--w_reg', default=0.05, type=float,
                        help='weight for regularization')

    parser.add_argument("--backbone", type=str, default="ToCo",
                        help="choose backone")
    parser.add_argument("--ignore_index", default=255, type=int, help="random index")
    parser.add_argument('--high_threshold', default=0.65, type=float,
                        help='high threshold')
    parser.add_argument('--low_threshold', default=0.3, type=float,
                        help='low threshold')
    parser.add_argument('--background_threshold', default=0.3, type=float,
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
            args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform,
            local_crop_size=args.local_crop_size,
        )
        total_size = len(train_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Split dataset
        train_subset, val_subset, test_subset = random_split(train_dataset, [train_size, val_size, test_size])
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
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

        print(f"Train size: {len(train_subset)} |Val size: {len(val_subset)} |Test size: {len(test_subset)}")

    # ---- Model ----

    model = choose_backbone(args.backbone)
    model.to(device)

    # ---- Define Loss & Optimizer ----
    avg_meter = AverageMeter('cls_loss', 'ptc_loss', 'ctc_loss', 'cls_loss_aux', 'seg_loss', 'cls_score', 'total_loss',
                             'reg_loss')

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    ncrops = 10

    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=1.0).cuda()

    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    param_groups = model.get_param_groups()

    optimizer = optim.AdamW(
        params=[
            {'params': param_groups[0], 'lr': args.lr},
            # norm layers
            {'params': param_groups[1], 'lr': args.lr},
            {'params': param_groups[2], 'lr': args.lr * 10},
            {'params': param_groups[3], 'lr': args.lr * 10},
        ],
        lr=args.lr)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(
    #             # params=[
    #             #     {'params': model.backbone.parameters(), 'lr': args.lr / 10},
    #             #     {'params': model.classifier.parameters(), 'lr': args.lr}
    #             # ],
    #             model.parameters(),
    #             lr=args.lr,
    #             momentum=args.momentum,
    #             weight_decay=args.weight_decay
    #             )
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
            # if cls_labels.ndim ==1:
            #     cls_labels=F.one_hot(cls_labels, num_classes=2).float()
            cls = cls[:, 1]
            cls_aux = cls_aux[:, 1]
            pos_weight = torch.tensor([1.0]).to(device)
            cls_loss = F.binary_cross_entropy_with_logits(
                cls,
                cls_labels.float(),
                pos_weight=pos_weight)

            cls_loss_aux = F.binary_cross_entropy_with_logits(
                cls_aux,
                cls_labels.float(),
                pos_weight=pos_weight)

            ctc_loss = CTC_loss(out_s, out_t, flags)

            valid_cam, _ = cam_to_label(
                cams.detach(),
                cls_labels=cls_labels,
                # img_box=None,
                ignore_mid=True,
                bkg_thre=args.background_threshold,
                high_thre=args.high_threshold,
                low_thre=args.low_threshold,
                ignore_index=args.ignore_index)

            valid_cam_aux, _ = cam_to_label(
                cams_aux.detach(),
                cls_labels=cls_labels,
                # img_box=None,
                ignore_mid=True,
                bkg_thre=args.background_threshold,
                high_thre=args.high_threshold,
                low_thre=args.low_threshold,
                ignore_index=args.ignore_index)

            if epoch <= 10:
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
            # print(f"Label min: {refined_pseudo_label.min().item()}, max: {refined_pseudo_label.max().item()}")
            # print(f"Number of classes in segs: {segs.size(1)}")
            # print(f"segs shape: {segs.shape}")

            segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

            seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

            reg_loss = get_energy_loss(
                img=images,
                logit=segs,
                label=refined_pseudo_label,
                # img_box=None,
                loss_layer=loss_layer)

            resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)

            pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_labels=cls_labels,
                                            # img_box=None,
                                            ignore_mid=False,
                                            bkg_thre=args.background_threshold,
                                            high_thre=args.high_threshold,
                                            low_thre=args.low_threshold,
                                            ignore_index=args.ignore_index)

            # torch.Size([8, 32, 32])
            # print("pseudo_label_aux",pseudo_label_aux.shape)

            aff_mask = label_to_aff_mask(pseudo_label_aux)

            ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
            # ptc_loss = get_ptc_loss(fmap, low_fmap)
            # Early training phase - Using only classification losses")
            # if epoch <= 5:
            #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.0 * seg_loss + 0.0 * reg_loss
            # #Middle training phase - Adding segmentation and regularization losses"
            # elif epoch <= 10:
            #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * ctc_loss + 0.1 * seg_loss + args.w_reg * reg_loss
            # # Final training phase - Using all losses
            # else:
            #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.5 * ctc_loss + 0.1 * seg_loss + args.w_reg * reg_loss
            # if epoch<=15:
            #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux
            # else:
            #     loss = 1.0 * cls_loss + 1.0 * cls_loss_aux+0.1 * seg_loss
            # else:
            #     loss=1.0 * cls_loss + 1.0 * cls_loss_aux + 0.1 * seg_loss + args.w_reg * reg_loss
            if epoch <= 10:
                # cls_loss_aux too dominant
                loss = 1.0 * cls_loss + 0.01 * cls_loss_aux
            else:
                seg_weight = min(0.1 * (epoch - 10), 1.0) if epoch > 10 else 0.0  # linear growth

                loss = 1.0 * cls_loss + 0.01 * cls_loss_aux + seg_weight * seg_loss

            cls_pred = (cls > 0).type(torch.int16)
            cls_score = metrics.f1_score(cls_labels.cpu().numpy().flatten(), cls_pred.cpu().numpy().flatten())

            avg_meter.add({
                'cls_loss': cls_loss.item(),
                'ptc_loss': ptc_loss.item(),
                'ctc_loss': ctc_loss.item(),
                'cls_loss_aux': cls_loss_aux.item(),
                'seg_loss': seg_loss.item(),
                'cls_score': cls_score,
                'reg_loss': reg_loss.item(),
                'total_loss': loss.item(),
            })

            loss.backward()
            # update weights
            optimizer.step()
            if args.lr_scheduler == 'poly':
                scheduler.step()

        print(f"Epoch: {epoch}/{args.num_epochs} "
              f"Loss: {avg_meter.get('total_loss'):.4f}, "
              f"Cls Loss: {avg_meter.get('cls_loss'):.4f}, "
              f"Seg Loss: {avg_meter.get('seg_loss'):.4f}, "
              f"Reg Loss: {avg_meter.get('reg_loss'):.4f}, "
              f"PTC Loss: {avg_meter.get('ptc_loss'):.4f}, "
              f"CTC Loss: {avg_meter.get('ctc_loss'):.4f}, "
              f"Cls Loss Aux: {avg_meter.get('cls_loss_aux'):.4f}, "
              f"Cls Score: {avg_meter.get('cls_score'):.4f}", flush=True)

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
    model.load_state_dict(torch.load(save_path))
    model.eval()
    avg_meter = AverageMeter('cls_loss', 'ptc_loss', 'ctc_loss', 'cls_loss_aux', 'seg_loss', 'cls_score', 'total_loss')
    preds, gts, cams, cams_aux = [], [], [], []
    with torch.no_grad():
        for i, (images, cls_labels, image_ids, masks) in enumerate(test_loader_1):
            images = images.to(device)
            masks = masks.to(device)
            cls_labels = cls_labels.to(device)
            if cls_labels.ndim == 1:
                cls_labels = F.one_hot(cls_labels, num_classes=2).float()
            inputs = F.interpolate(images, size=[args.crop_size, args.crop_size],
                                   mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs, )
            cls_pred = (cls > 0).type(torch.int16)
            # print(f"cls_labels shape: {cls_labels.shape}, cls_pred shape: {cls_pred.shape}")
            _f1 = metrics.f1_score(cls_labels.cpu().numpy().reshape(-1), cls_pred.cpu().numpy().reshape(-1))
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            # print("_cams:", np.unique(_cams.cpu().numpy().astype(np.int16)))
            # print("_cams_aux:", np.unique(_cams_aux.cpu().numpy().astype(np.int16)))
            resized_cam = F.interpolate(_cams, size=masks.shape[1:], mode='bilinear', align_corners=False)

            cam_label = cam_to_label(resized_cam, cls_labels,
                                     bkg_thre=args.background_threshold,
                                     high_thre=args.high_threshold, low_thre=args.low_threshold,
                                     ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=masks.shape[1:],
                                            mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_labels,
                                         bkg_thre=args.background_threshold, high_thre=args.high_threshold,
                                         low_thre=args.low_threshold, ignore_index=args.ignore_index)

            # cls_pred = (cls > 0).type(torch.int16)
            # _f1 = metrics.f1_score(cls_labels.cpu().numpy().reshape(-1), cls_pred.cpu().numpy().reshape(-1))
            # avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(masks.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

        # print("gts:", np.unique(gts))
        # print("preds:", np.unique(preds))
        # print("cams:", np.unique(cams))
        # print("cams_aux:", np.unique(cams_aux))
        cls_score = avg_meter.pop('cls_score')
        seg_score = scores(gts, preds, num_classes=2)
        cam_score = scores(gts, cams, num_classes=2)
        cam_aux_score = scores(gts, cams_aux, num_classes=2)

    print("\nTest Results:")
    print(f"Cls Score: {cls_score:.4f}")
    print(f"Seg Score: {seg_score}")
    print(f"Cam Score: {cam_score}")
    print(f"Cam Aux Score: {cam_aux_score}")

