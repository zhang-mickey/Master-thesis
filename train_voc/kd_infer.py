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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import random_split, Subset
import pydensecrf.densecrf as dcrf
from sklearn.manifold import TSNE

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
from post_processing.cam_compare import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.dataset.cropDataset import *
from lib.utils.augmentation import *
from train_voc.dataset import *


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,
                                                                      "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
                        help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
                        help="Path to the image dataset folder")

    parser.add_argument("--non_smoke_image_folder", type=str, default=os.path.join(project_root, "lib/dataset/frames/"),
                        help="Path to the non-smoke image dataset folder")

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

    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/pseudo_labels"),
                        help="Path to save the pseudo labels")

    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root,
                                                                          "result/cam"), help="Path to save the cam")

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

    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    parser.add_argument("--num_epochs", type=int, default=3, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    # parser.add_argument("--save_model_path", type=str, default=os.path.join(project_root,"final_model/model_classification_without900.pth"), help="Path to save the trained model")
    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification_kd.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--backbone", type=str, default="vit_s",
                        help="choose backone")

    # parser.add_argument("--backbone", type=str, default="resnet50_raw",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="resnet50",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="resnet101",
    # help="choose backone")

    parser.add_argument('--manual_seed', default=42, type=int, help='Manually set random seed')

    # infer
    parser.add_argument("--threshold", type=float, default=0.3, help="threshold to pesudo label")

    parser.add_argument("--voc12", type=bool, default=True, help="use VOC12 dataset or not")
    parser.add_argument("--voc12_root", type=str, default=os.path.join(project_root, "VOCdevkit/VOC2012/"),
                        help="path to VOC2012 dataset root directory")
    parser.add_argument("--voc12_list", type=str,
                        default=os.path.join(project_root, "VOCdevkit/VOC2012/ImageSets/Segmentation/"),
                        help="path to VOC2012 dataset list directory")
    parser.add_argument("--voc12_split", type=str, default="train",
                        help="dataset split to use (train, val, test)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform, mask_transform = get_transforms(args.img_size)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)

    if args.use_crop:
        train_dataset = VOC12SegDataset(
            root_dir=args.voc12_root,
            name_list_dir=args.voc12_list,
            split=args.voc12_split,
            stage='train',
            crop_size=args.crop_size,
            aug=True  # Enable augmentation for training
        )
        # For testing/validation
        test_dataset = VOC12SegDataset(
            root_dir=args.voc12_root,
            name_list_dir=args.voc12_list,
            split='val',  # Use validation split for testing
            stage='val',
            crop_size=args.crop_size,
            aug=False  # No augmentation for testing
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"VOC12 Training Dataset loaded: {len(train_dataset)} images in total")
        print(f"VOC12 Test Dataset loaded: {len(test_dataset)} images in total")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"best_{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )

    save_resnet_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"resnet50_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )

    model = choose_backbone(args.backbone)
    model_resnet = choose_backbone("resnet50")
    model = model.to(device)

    resnet_target_layers = [model_resnet.layer4[-1]]  # Last layer of layer4
    vit_target_layers = [model.blocks[-1].norm1]  # Last transformer block

    # target_layers = [
    # model.blocks[-5].norm1,  #
    # model.blocks[-4].norm1,  #
    # model.blocks[-2].norm1, ]#

    model.load_state_dict(torch.load(save_path))
    model.eval()

    save_PCM_cam_path = os.path.join(
        os.path.dirname(args.save_cam_path),
        f"{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}_PCM"
    )

    save_cam_path = os.path.join(
        os.path.dirname(args.save_cam_path),
        f"kd_{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}_kd"
    )
    # save_resnet_cam_path = os.path.join(
    #         os.path.dirname(args.save_cam_path),
    #         f"kd_resnet50_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}_kd"
    #     )
    generate_cam_for_dataset(
        dataloader=train_loader,
        model=model,
        target_layers=vit_target_layers,
        save_dir=save_cam_path,
    )

    # generate_cam_for_dataset(
    #     dataloader=train_loader,
    #     model=model_resnet,
    #     target_layers=resnet_target_layers,
    #     save_dir=save_resnet_cam_path,
    # )
    # if PCM, no need target_layer
    # generate_PCM_cam(
    #     dataloader=train_loader,
    #     model=model,
    #     # target_layers=target_layers,
    #     save_dir=save_PCM_cam_path,
    # )

    # generate_PCM_pseudo_labels(
    #     dataloader=train_loader,
    #     model=model,
    #     save_dir=save_PCM_cam_path,
    #     threshold=args.threshold
    # )

    # sliding_window_patch_cam_generate(
    #     dataloader=train_loader,
    #     model=model,
    #     target_layers=target_layers,
    #     save_dir=save_cam_path,
    # )

    # save_crop_cam_path = os.path.join(
    #         os.path.dirname(args.save_cam_path),
    #         f"{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}_crop"
    #     )

    # generate_crop_cam_for_dataset(
    #     dataloader=train_loader,
    #     model=model,
    #     target_layers=target_layers,
    #     save_dir=save_crop_cam_path,
    #     aug=True
    # )

    # sliding_window_cam_generate(
    #     dataloader=train_loader,
    #     model=model,
    #     target_layers=target_layers,
    #     save_dir=save_crop_cam_path,
    #     aug=True
    # )

    save_pseudo_labels_path = os.path.join(
        os.path.dirname(args.save_pseudo_labels_path),
        f"kd_{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_pseudo_labels_path)}_kd"
    )
    save_resnet_pseudo_labels_path = os.path.join(
        os.path.dirname(args.save_pseudo_labels_path),
        f"kd_resnet50_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_pseudo_labels_path)}_kd"
    )

    generate_pseudo_labels(
        dataloader=train_loader,
        model=model,
        target_layers=vit_target_layers,
        save_dir=save_pseudo_labels_path,
        threshold=args.threshold
    )
    # generate_pseudo_labels(
    #     dataloader=train_loader,
    #     model=model_resnet,
    #     target_layers=resnet_target_layers,
    #     save_dir=save_resnet_pseudo_labels_path,
    #     threshold=args.threshold
    # )
