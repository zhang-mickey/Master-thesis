import argparse
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from lib.dataset.cropDataset import *


def parse_args():
    parser = argparse.ArgumentParser(description="SAM fine tune")
    parser.add_argument("--crop_smoke_image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images/"),
                        help="Path to the cropped smoke image dataset folder")
    parser.add_argument("--crop_mask_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks/"),
                        help="Path to the cropped image dataset mask folder")
    parser.add_argument("--model_type", type=str, default="vit_h",
                        choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/sam_vit_h_4b8939.pth",
                        help="Pretrained weights path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()


class SAMFineTuner(nn.Module):
    def __init__(self, model_type="vit_h", checkpoint="pretrained/sam_vit_h_4b8939.pth",
                 device="cuda", lr=1e-4):
        super().__init__()

        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.model.to(device)

        # 仅微调部分参数
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False

        # 解冻mask decoder
        for param in self.model.mask_decoder.parameters():
            param.requires_grad = True

        # 创建可学习的任务提示嵌入
        self.task_embedding = nn.Parameter(
            torch.randn(1, 1, 256, device=device) * 0.02  # 增加维度 [1, 1, 256]
        )
        # 固定位置编码（PE）
        self.fixed_pe = self.model.prompt_encoder.get_dense_pe().to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.mask_decoder.parameters()},
            {'params': self.task_embedding}
        ], lr=lr)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, images):
        # 生成图像嵌入
        images = images.to(self.device)
        image_embeddings = self.model.image_encoder(images)
        batch_size = images.shape[0]

        # 准备任务提示 [batch_size, 1, 256]
        sparse_embeddings = self.task_embedding.expand(batch_size, -1, -1)

        # 准备位置编码 (PE)，使用原始维度
        fixed_pe = self.fixed_pe.to(images.device)

        # 准备密集提示嵌入 - 全零 [batch_size, C, H, W]
        dense_prompt_embeddings = torch.zeros(
            batch_size,
            image_embeddings.shape[1],
            image_embeddings.shape[2],
            image_embeddings.shape[3],
            device=self.device
        )

        # 预测掩码
        mask_pred, _ = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=fixed_pe,  # 使用原始位置编码
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )

        return mask_pred

    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for i, (images, labels, _, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            gt_masks = masks.to(self.device)

            mask_pred = self(images)

            # 调整预测掩码的尺寸以匹配目标掩码(1024x1024)
            mask_pred_resized = F.interpolate(
                mask_pred,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )

            # 计算损失 - 确保尺寸匹配
            loss = self.criterion(mask_pred_resized, gt_masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    def save_model(self, save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'task_embedding': self.task_embedding
        }, save_path)
        print(f"Model saved to {save_path}")

    def evaluate(self, dataloader, threshold=0.3):
        self.model.eval()
        total_iou = 0.0
        total_dice = 0.0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for i, (images, labels, _, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                gt_masks = masks.to(self.device)

                mask_pred = self(images)

                mask_pred_resized = F.interpolate(
                    mask_pred,
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                )

                prob_mask = torch.sigmoid(mask_pred_resized)
                binary_mask = (prob_mask > threshold).float()

                # 计算每个样本的IoU和Dice
                for j in range(images.shape[0]):
                    pred_mask = binary_mask[j].squeeze().cpu().numpy().astype(np.uint8)
                    gt_mask = gt_masks[j].squeeze().cpu().numpy().astype(np.uint8)

                    # 避免空掩码
                    if np.max(pred_mask) == 0 and np.max(gt_mask) == 0:
                        iou = 1.0
                        dice = 1.0
                    else:
                        # 计算交集和并集
                        intersection = np.logical_and(pred_mask, gt_mask).sum()
                        union = np.logical_or(pred_mask, gt_mask).sum()

                        iou = intersection / union if union > 0 else 0.0
                        dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)

                    total_iou += iou
                    total_dice += dice
                    total_samples += 1

                progress_bar.set_postfix({"IoU": total_iou / total_samples, "Dice": total_dice / total_samples})

        mean_iou = total_iou / total_samples if total_samples > 0 else 0.0
        mean_dice = total_dice / total_samples if total_samples > 0 else 0.0

        return mean_iou, mean_dice


def main():
    args = parse_args()
    print("root_dir:", project_root)

    # 创建数据集
    dataset = CropDataset(
        args.crop_smoke_image_folder,
        args.crop_mask_folder,
    )

    print(f"数据集大小: {len(dataset)}")

    # 划分数据集
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    print(f"测试集大小: {len(test_subset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化微调器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    fine_tuner = SAMFineTuner(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        device=device,
        lr=args.lr
    )

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"模型将保存在: {args.save_dir}")

    # 训练循环
    best_iou = 0.0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = fine_tuner.train_epoch(train_loader)

        # 验证
        val_iou, val_dice = fine_tuner.evaluate(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            fine_tuner.save_model(best_model_path)
            print(f"保存最佳模型，IoU: {best_iou:.4f}")

    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(best_model_path, map_location=fine_tuner.device)
    fine_tuner.model.load_state_dict(checkpoint['model_state_dict'])
    fine_tuner.task_embedding = checkpoint['task_embedding'].to(fine_tuner.device)

    test_iou, test_dice = fine_tuner.evaluate(test_loader)
    print(f"\n最终测试结果 - IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()