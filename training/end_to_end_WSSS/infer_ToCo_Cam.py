from this import d
import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import imageio
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

    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/pseudo_labels"),
                        help="Path to save the pseudo labels")

    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root, "result/cam"),
                        help="Path to save the cam")

    parser.add_argument("--save_visualization_path", type=str,
                        default=os.path.join(project_root, "result/visualization"),
                        help="Path to save the cam")

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
    parser.add_argument("--crop_size", default=512, type=int, help="crop_size for global view")

    parser.add_argument('--w_reg', default=0.05, type=float,
                        help='weight for regularization')

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

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

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

    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )
    save_cam_path = os.path.join(
        os.path.dirname(args.save_cam_path),
        f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}"
    )
    # ---- Load the trained model for testing ----
    model.load_state_dict(torch.load(save_path))
    model.eval()
    color_map = plt.get_cmap('jet')

    avg_meter = AverageMeter('cls_loss', 'ptc_loss', 'ctc_loss', 'cls_loss_aux', 'seg_loss', 'cls_score', 'total_loss')
    with torch.no_grad():
        model.cuda()
        gts, cams, cams_aux = [], [], []
        for i, (images, cls_labels, image_ids, masks) in enumerate(train_loader_1):
            images = images.to(device)
            img_denorm = denormalize_img2(images)

            masks = masks.to(device)
            cls_labels = cls_labels.to(device)
            if cls_labels.ndim == 1:
                cls_labels = F.one_hot(cls_labels, num_classes=2).float()
            inputs = F.interpolate(images, size=[args.crop_size, args.crop_size],
                                   mode='bilinear', align_corners=False)

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=masks.shape[1:],
                                        mode='bilinear', align_corners=False)
            resized_cam_aux = F.interpolate(_cams_aux, size=masks.shape[1:],
                                            mode='bilinear', align_corners=False)

            cam_label = cam_to_label(resized_cam, cls_labels,
                                     bkg_thre=args.background_threshold)

            cam_aux_label = cam_to_label(resized_cam_aux, cls_labels,
                                         bkg_thre=args.background_threshold)

            resized_cam = get_valid_cam(resized_cam, cls_labels)
            resized_cam_aux = get_valid_cam(resized_cam_aux, cls_labels)

            cam_np = torch.max(resized_cam[0], dim=0)[0].cpu().numpy()
            cam_aux_np = torch.max(resized_cam_aux[0], dim=0)[0].cpu().numpy()

            cam_rgb = color_map(cam_np)[:, :, :3] * 255
            cam_aux_rgb = color_map(cam_aux_np)[:, :, :3] * 255

            alpha = 0.6
            cam_rgb = alpha * cam_rgb + (1 - alpha) * img
            cam_aux_rgb = alpha * cam_aux_rgb + (1 - alpha) * img

            img_id = image_ids[0]
            imageio.imsave(os.path.join(save_cam_path, f"cam_{img_id}.png"), cam_rgb.astye(np.uint8))

            imageio.imsave(os.path.join(save_cam_path, f"aux_cam_{img_id}.png"), cam_aux_rgb.astye(np.uint8))

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(masks.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_aux_label.cpu().numpy().astype(np.int16))

        print("gts:", np.unique(gts))
        print("cams:", np.unique(cams))
        print("cams_aux:", np.unique(cams_aux))

        cam_score = scores(gts, cams, num_classes=2)
        cam_aux_score = scores(gts, cams_aux, num_classes=2)

    print(f"Cam Score: {cam_score}")
    print(f"Cam Aux Score: {cam_aux_score}")


