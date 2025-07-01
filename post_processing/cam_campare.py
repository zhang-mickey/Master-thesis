import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import torchvision.transforms as T
import os, argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import random_split, Subset
import pydensecrf.densecrf as dcrf
from sklearn.manifold import TSNE
from pytorch_grad_cam import ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.dataset.WeaklyDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from post_processing.single_image_infer import *
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

    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "final_model/model_classification.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--backbone", type=str, default="vit_s",
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

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform, mask_transform = get_transforms(args.img_size)

    single_image_path = "frames/manual_positive/1NKYOpxE90A-2_frame_000021.jpg"
    # single_image_path="frames/manual_positive/On1da9YV4U8-1_frame2.jpg"

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)

    base_model = choose_backbone(args.backbone)
    base_model.to(device)

    our_model = choose_backbone(args.backbone)
    our_model.to(device)

    baseline_model_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"transformer_{args.CAM_type}_{args.num_epochs}_model_classification_without900.pth"
    )

    our_model_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{args.CAM_type}_3_model_classification_kd.pth"
    )

    base_model.load_state_dict(torch.load(baseline_model_path))
    our_model.load_state_dict(torch.load(our_model_path))

    target_layers_base = [base_model.blocks[-1].norm1]
    target_layers_base_fusion = [base_model.blocks[-5].norm1, base_model.blocks[-4].norm1, base_model.blocks[-2].norm1,
                                 base_model.blocks[-1].norm1]

    target_layers_our = [our_model.blocks[-1].norm1]
    target_layers_our_fusion = [our_model.blocks[-5].norm1, our_model.blocks[-4].norm1, our_model.blocks[-2].norm1,
                                our_model.blocks[-1].norm1]

    base_model.eval()
    our_model.eval()

    save_cam_path = os.path.join(
        os.path.dirname(args.save_cam_path),
        f"{args.backbone}_{os.path.basename(args.save_cam_path)}"
    )

    raw_image = cv2.imread(single_image_path)[:, :, ::-1]  # BGR to RGB
    raw_image = cv2.resize(raw_image, (args.img_size, args.img_size))
    rgb_float = raw_image.astype(np.float32) / 255.0

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(Image.fromarray(raw_image)).unsqueeze(0).to(device)

    cam_image_base, grayscale_cam_base, pred_class_base = get_cam_for_image(image_tensor, base_model,
                                                                            target_layers_base)

    cam_image_our, grayscale_cam_our, pred_class_our = get_cam_for_image(image_tensor, our_model, target_layers_our)

    cam_image_base_fusion, _, _ = get_cam_for_image(image_tensor, base_model, target_layers_base)

    cam_image_our_fusion, _, _ = get_cam_for_image(image_tensor, our_model, target_layers_our_fusion)

    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    axs[0].imshow(raw_image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Baseline CAM
    axs[1].imshow(cam_image_base)
    axs[1].set_title("Baseline CAM")
    axs[1].axis('off')

    # Our CAM
    axs[2].imshow(cam_image_our)
    axs[2].set_title("Our CAM")
    axs[2].axis('off')
    # Our CAM+post_processing
    axs[3].imshow(cam_image_our_fusion)
    axs[3].set_title("Our CAM+post processing")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig("1.png")
    plt.close()