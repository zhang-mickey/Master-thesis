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
from post_processing.inference import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.dataset.cropDataset import *
from lib.utils.augmentation import *


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

    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification_without900.pth"),
                        help="Path to save the trained model")

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

    parser.add_argument("--num_epochs", type=int, default=2, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--backbone", type=str, default="transformer",
                        help="choose backone")
    # parser.add_argument("--backbone", type=str, default="resnet101",
    # help="choose backone")
    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    # help="choose backone")
    # parser.add_argument("--backbone", type=str, default="deeplabv3plus_resnet101", help="choose backone")

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    # infer
    parser.add_argument("--threshold", type=float, default=0.3, help="threshold to pesudo label")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform, mask_transform = get_transforms(args.img_size)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)

    # train_dataset = SmokeDataset(args.json_path, args.image_folder,
    #                         args.smoke5k_path,args.Rise_path,
    #                         transform=image_transform,
    #                         mask_transform=mask_transform,
    #                         image_ids=train_ids,
    #                         return_both=args.augmentation,
    #                         smoke5k=args.smoke5k,Rise=args.Rise)
    train_dataset = CropDataset(
        image_dir=args.crop_smoke_image_folder,
        mask_dir=args.crop_mask_folder,
        # non_smoke_dir=args.crop_non_smoke_folder,
        # ijmond_positive_dir=args.Dutch_positive_path,
        # ijmond_negative_dir=args.Dutch_negative_path,
        transform=image_transform,
        mask_transform=mask_transform,
        img_size=(args.crop_size, args.crop_size),
        backbone=args.backbone
    )

    print(f"Total train samples: {len(train_dataset)}")

    total_size = len(train_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_size = int(0.95 * total_size)
    test_size = total_size - train_size

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(train_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = choose_backbone(args.backbone)
    model = model.to(device)

    model.load_state_dict(torch.load(save_path))
    model.eval()

    fg_features = []  # 用于存储所有前景特征
    bg_features = []  # 用于存储所有背景特征

    with torch.no_grad():
        for batch_idx, (images, labels, _, mask) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)  # 确保mask也在设备上

            outputs, _, feature_maps = model(images)
            B, C, H, W = feature_maps.shape

            for i in range(B):
                img_features = feature_maps[i]  # [C, H, W]

                # 调整mask大小以匹配特征图空间尺寸
                # 假设原mask尺寸是[1, H_orig, W_orig]
                cur_mask = mask[i].float()  # [1, H_orig, W_orig] or [H_orig, W_orig]
                if len(cur_mask.shape) == 3:
                    cur_mask = cur_mask[0]  # 如果是 [1, H, W]，去掉通道维度

                # 调整到特征图空间尺寸
                if cur_mask.shape != (H, W):
                    cur_mask = F.interpolate(
                        cur_mask.unsqueeze(0).unsqueeze(0),  # 添加批次和通道维度
                        size=(H, W),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)  # 移除添加的维度

                # 创建二值掩码
                fg_mask = cur_mask > 0.5  # [H, W]，True表示烟雾区域
                bg_mask = ~fg_mask  # [H, W]，True表示背景区域

                # 展平特征
                flat_features = img_features.view(C, H * W)  # [C, H*W]

                # 收集前景特征
                if torch.any(fg_mask):
                    fg_pixels = flat_features[:, fg_mask.flatten()]
                    fg_features.append(fg_pixels.cpu())

                # 收集背景特征
                if torch.any(bg_mask):
                    bg_pixels = flat_features[:, bg_mask.flatten()]
                    bg_features.append(bg_pixels.cpu())

    if fg_features:
        all_fg = torch.cat(fg_features, dim=1)  # [C, total_fg_pixels]
        fg_prototype = all_fg.mean(dim=1)  # [C]
        fg_prototype = F.normalize(fg_prototype, p=2, dim=0)

    if bg_features:
        all_bg = torch.cat(bg_features, dim=1)  # [C, total_bg_pixels]
        bg_prototype = all_bg.mean(dim=1)  # [C]
        bg_prototype = F.normalize(bg_prototype, p=2, dim=0)

        # 计算和显示相似度
    if fg_prototype is not None and bg_prototype is not None:
        similarity = F.cosine_similarity(
            fg_prototype.unsqueeze(0),
            bg_prototype.unsqueeze(0),
            dim=1
        ).item()
        print(f"FG-BG Similarity: {similarity:.4f}")
    else:
        print("Warning: Missing prototypes to compute similarity")
    # 假设 fg_features 和 bg_features 是 list，每个元素是 shape [C, N_i] 的 tensor
    # 合并所有特征

    fg_label = 1
    bg_label = 0

    if fg_features and bg_features:
        all_fg = torch.cat(fg_features, dim=1).T  # [total_fg_pixels, C]
        all_bg = torch.cat(bg_features, dim=1).T  # [total_bg_pixels, C]

        # 合并前景背景特征
        features = torch.cat([all_fg, all_bg], dim=0)  # [N_fg+N_bg, C]
        labels = torch.tensor([fg_label] * all_fg.shape[0] + [bg_label] * all_bg.shape[0])

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_2d = tsne.fit_transform(features.cpu().numpy())

        # 可视化
        plt.figure(figsize=(8, 6))
        plt.scatter(features_2d[labels == fg_label, 0], features_2d[labels == fg_label, 1], c='red', label='Foreground',
                    alpha=0.6)
        plt.scatter(features_2d[labels == bg_label, 0], features_2d[labels == bg_label, 1], c='blue',
                    label='Background', alpha=0.6)
        plt.title("t-SNE of Foreground and Background Features")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("tsne_fg_bg.png")
        plt.close()