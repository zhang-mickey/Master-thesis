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
from torch.utils.data import random_split
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
from inference.inference import *
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
                        default=os.path.join(project_root, "final_model/model_classification.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=False, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")
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

    parser.add_argument("--num_epochs", type=int, default=10, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")

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
    # train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

    # print(f"Smoke Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    # non_smoke_train, non_smoke_val, non_smoke_test = split_non_smoke_dataset(args.non_smoke_image_folder)
    # print(f"Non-smoke Dataset split: Train={len(non_smoke_train)}, Val={len(non_smoke_val)}, Test={len(non_smoke_test)}")

    if args.use_crop:
        train_dataset = CropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform
        )
        # print(f"Train size: {len(train_dataset)}")

        # Suppose you want 80% training and 20% testing
        total_size = len(train_dataset)
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size

        # Split dataset
        train_subset, test_subset = random_split(train_dataset, [train_size, test_size])
        print(f"Train size: {len(train_subset)} | Test size: {len(test_subset)}")
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = choose_backbone(args.backbone)
    model = model.to(device)

    if args.backbone == "resnet101":
        target_layers = [model.layer4[-1]]  # Last layer of layer4
    elif args.backbone == "transformer":
        target_layers = [model.blocks[-1].norm1]  # Last transformer block
    elif args.backbone == "mix_transformer":
        print(dir(model))
        print("-------------------")
        target_layers = [model.norm4]  # Last transformer block
        print(target_layers)
    else:
        target_layers = [list(model.children())[-3]]

    model.load_state_dict(torch.load(save_path))
    model.eval()
    # t-SNE
    all_layers_cls = {i: [] for i in range(13)}
    all_layers_cls_aug = {i: [] for i in range(13)}

    labels_list = []
    with torch.no_grad():
        for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            augmented_images = augment_batch(images)
            _, _, cls_embeddings_all_layers_aug = model.forward_features(augmented_images)
            _, _, cls_embeddings_all_layers = model.forward_features(images)  # [B, D]

            for layer_idx, cls_emb in enumerate(cls_embeddings_all_layers):
                all_layers_cls[layer_idx].append(cls_emb.cpu().numpy())
            for layer_idx, cls_emb in enumerate(cls_embeddings_all_layers_aug):
                all_layers_cls_aug[layer_idx].append(cls_emb.cpu().numpy())

            labels_list.append(labels.cpu().numpy())  # Ensure labels are 0/1

    for layer_idx in all_layers_cls:
        all_layers_cls[layer_idx] = np.concatenate(all_layers_cls[layer_idx], axis=0)
    for layer_idx in all_layers_cls_aug:
        all_layers_cls_aug[layer_idx] = np.concatenate(all_layers_cls_aug[layer_idx], axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    selected_layers = [2, 7, 11]

    # For each layer, compute similarity between original and augmented CLS embeddings
    for layer_idx in selected_layers:
        orig_embs = all_layers_cls[layer_idx]  # [N, D]
        aug_embs = all_layers_cls_aug[layer_idx]

        # Compute pairwise cosine similarity
        similarities = np.einsum('nd,nd->n', orig_embs, aug_embs) / (
                np.linalg.norm(orig_embs, axis=1) * np.linalg.norm(aug_embs, axis=1)
        )
        print(f"Layer {layer_idx}: Avg similarity = {similarities.mean():.3f}")

    plt.figure(figsize=(20, 4))
    for i, layer_idx in enumerate(selected_layers):
        embeddings = all_layers_cls[layer_idx]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.subplot(1, len(selected_layers), i + 1)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=10)
        plt.title(f"Layer {layer_idx}")
        plt.axis('off')
    plt.colorbar(ticks=[0, 1], label='Class')
    # plt.title("t-SNE of CLS Embeddings (Binary Classification)")
    plt.savefig("tsne.png")
    plt.close()

    plt.figure(figsize=(20, 4))
    for i, layer_idx in enumerate(selected_layers):
        embeddings = all_layers_cls_aug[layer_idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.subplot(1, len(selected_layers), i + 1)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=10)
        plt.title(f"Layer {layer_idx}")
        plt.axis('off')
    plt.colorbar(ticks=[0, 1], label='Class')
    # plt.title("t-SNE of CLS Embeddings (Binary Classification)")
    plt.savefig("tsne_aug.png")
    plt.close()