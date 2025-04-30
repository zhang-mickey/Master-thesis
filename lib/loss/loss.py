import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import math

sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
from bilateralfilter import bilateralfilter, bilateralfilter_batch


def plot_loss_accuracy(loss_history, acc_history, save_path=None):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss', color='tab:red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Training Accuracy', color='tab:green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()

    # Binary segmentation (1 class + background)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


# one positive one negative per anchor
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


class MaxMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(self.margin + pos_dist - neg_dist)  # Ensure positive margin
        return loss.mean()


# one positive multiple negatives per anchor
class NPairsLoss(nn.Module):
    def __init__(self):
        super(NPairsLoss, self).__init__()

    def forward(self, anchor, positive, negatives):
        """
        - anchor: (B, D) tensor of anchor embeddings
        - positive: (B, D) tensor of positive embeddings
        - negatives: (B, N, D) tensor of negative embeddings (N negatives per sample)
        """
        batch_size = anchor.shape[0]

        # Compute similarity scores
        pos_sim = F.cosine_similarity(anchor, positive)  # (B,)
        neg_sim = torch.cat([F.cosine_similarity(anchor, neg) for neg in negatives], dim=0)  # (B*N,)

        # LogSumExp trick for numerical stability
        log_prob = torch.logsumexp(neg_sim.view(batch_size, -1), dim=1)  # (B,)
        loss = -pos_sim + log_prob  # N-Pairs Loss

        return loss.mean()


def pixel_contrastive_loss(features, masks, temperature=0.1, max_samples=256):
    """
    Compute contrastive loss at the pixel level.
    Args:
        features: Feature maps from the model (B, C, H, W)
        masks: Pseudo-label masks (B, H, W)
        temperature: Temperature scaling for contrastive loss.
    """

    B, C, H, W = features.shape
    features = F.normalize(features, p=2, dim=1)  # L2 normalize features

    loss = 0
    count = 0

    for b in range(B):
        mask = masks[b]  # (H, W)

        H_mask, W_mask = masks.shape[1:]  # 512, 512
        feature_map = features[b]  # (C, H, W)
        # If masks were originally 512×512 but feature_map is 128×128,
        # every coordinate in masks needs to be downscaled by 128/512 = 0.25
        # to fit within feature_map.
        H_feat, W_feat = feature_map.shape[1:]  # 128, 128

        # print("Mask shape:", mask.shape)
        scale_x = W_feat / W_mask
        scale_y = H_feat / H_mask

        pos_indices = torch.nonzero(mask == 1, as_tuple=False)  # Foreground pixels
        neg_indices = torch.nonzero(mask == 0, as_tuple=False)  # Background pixels

        pos_indices[:, 0] = (pos_indices[:, 0] * scale_y).long()
        pos_indices[:, 1] = (pos_indices[:, 1] * scale_x).long()

        neg_indices[:, 0] = (neg_indices[:, 0] * scale_y).long()
        neg_indices[:, 1] = (neg_indices[:, 1] * scale_x).long()
        # print("Feature map shape:", feature_map.shape) #[1,128,128]
        # print("Positive indices:", pos_indices)

        # sample a subset of positive and negative features
        if len(pos_indices) > max_samples:
            pos_indices = pos_indices[torch.randperm(len(pos_indices))[:max_samples]]
        if len(neg_indices) > max_samples:
            neg_indices = neg_indices[torch.randperm(len(neg_indices))[:max_samples]]

        if len(pos_indices) > 1 and len(neg_indices) > 1:
            pos_features = feature_map[:, pos_indices[:, 0], pos_indices[:, 1]]  # (C, N_pos)
            neg_features = feature_map[:, neg_indices[:, 0], neg_indices[:, 1]]  # (C, N_neg)

            # ✅ **Avoid large matrix multiplication by limiting pairs** ✅
            pos_sim = torch.exp(torch.mm(pos_features.T, pos_features) / temperature)
            neg_sim = torch.exp(torch.mm(pos_features.T, neg_features) / temperature)

            pos_loss = -torch.log(pos_sim.diag() / (pos_sim.sum() + neg_sim.sum()))
            loss += pos_loss.mean()
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=features.device)


# def get_seg_loss(pred, label, ignore_index=255):

#     bg_label = label.clone()
#     bg_label[label!=0] = ignore_index
#     bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
#     fg_label = label.clone()
#     fg_label[label==0] = ignore_index
#     fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

#     return (bg_loss + fg_loss) * 0.5

def get_seg_loss(pred, label, ignore_index=255):
    # print('label unique values:', label.unique())
    label = label.clone()
    label[label == 2] = 1
    label[~((label == 0) | (label == 1) | (label == ignore_index))] = ignore_index
    loss = F.cross_entropy(pred, label.long(), ignore_index=ignore_index)
    return loss


# def get_seg_loss(pred, label):
#     # pred: (B,1,H,W)
#     # label: (B,H,W) 0 or 1
#     pred = pred.squeeze(1)  # (B,H,W)
#     loss = F.binary_cross_entropy_with_logits(pred, label.float())
#     return loss


# Patch Token Contrast loss
# encourages feature similarity within the same class (foreground or background)
# and feature dissimilarity between different classes.
def get_masked_ptc_loss(inputs, mask, ignore_index=255):
    b, c, h, w = inputs.shape

    inputs = inputs.reshape(b, c, h * w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1, 2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)
    valid_mask = (mask != ignore_index)
    pos_mask = (mask == 1) & valid_mask
    neg_mask = (mask == 0) & valid_mask
    # only compute the loss for foreground and background
    loss = 0.5 * (1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum() + 1)) + 0.5 * torch.sum(
        neg_mask * inputs_cos) / (neg_mask.sum() + 1)
    return loss


def get_energy_loss(img, logit, label,
                    # img_box,
                    loss_layer,
                    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.ones_like(pred_prob[:, 0, ...])

    _img = torch.zeros_like(img)
    _img[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
    _img[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
    _img[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1))

    return loss.cuda()


class DenseEnergyLossFunction(Function):

    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)

        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)

        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2 * grad_output * torch.from_numpy(ctx.AS) / ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None


class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations, ROIs, seg_label, valid_mask=None):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images, scale_factor=self.scale_factor,
                                      recompute_scale_factor=True)

        scaled_segs = F.interpolate(segmentations, scale_factor=self.scale_factor,
                                    mode='bilinear', align_corners=False, recompute_scale_factor=True)

        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1), scale_factor=self.scale_factor,
                                    recompute_scale_factor=True).squeeze(1)

        scaled_seg_label = F.interpolate(seg_label, scale_factor=self.scale_factor,
                                         mode='nearest', recompute_scale_factor=True)

        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight * DenseEnergyLossFunction.apply(
            scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy * self.scale_factor, scaled_ROIs, unlabel_region)

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0, ):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

    def forward(self, student_output, teacher_output, flags):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        b = flags.shape[0]

        student_out = student_output.reshape(self.ncrops, b, -1).permute(1, 0, 2)
        teacher_out = teacher_output.reshape(2, b, -1).permute(1, 0, 2)

        logits = torch.matmul(teacher_out, student_out.permute(0, 2, 1))
        logits = torch.exp(logits / self.temp)

        total_loss = 0
        for i in range(b):
            neg_logits = logits[i, :, flags[i] == 0]
            pos_inds = torch.nonzero(flags[i])[:, 0]
            loss = 0

            for j in pos_inds:
                pos_logit = logits[i, :, j]
                loss += -torch.log((pos_logit) / (pos_logit + neg_logits.sum(dim=1) + 1e-4))
            else:
                loss += -torch.log((1) / (1 + neg_logits.sum(dim=1) + 1e-4))

            total_loss += loss.sum() / 2 / (pos_inds.shape[0] + 1e-4)

        total_loss = total_loss / b

        return total_loss


def attention_entropy_loss(attn_weights):
    """
    Args:
        attn_weights: tensor of shape [B, H, N, N] (batch, heads, num_tokens, num_tokens)
                      Usually from transformer blocks' attention outputs
    Returns:
        Scalar loss encouraging compact attention
    """
    B, H, N, _ = attn_weights.shape

    # Take attention from CLS token to image patches only
    cls_attn = attn_weights[:, :, 0, 1:]  # [B, H, N_patches]

    # Normalize across heads & tokens
    cls_attn = cls_attn.reshape(B, -1)  # Flatten heads + patches → [B, H*N]
    cls_attn = F.softmax(cls_attn, dim=-1)

    # Entropy: H(p) = -sum(p * log(p + eps))
    eps = 1e-8
    entropy = -torch.sum(cls_attn * torch.log(cls_attn + eps), dim=-1).mean()

    return entropy


def resize_dcp_to_patches(dcp_batch, target_num_patches=255):
    """
    Resize DCP maps to match number of image patches used by vision transformer.

    Args:
        dcp_batch: NumPy array of shape [B, H, W]
        target_num_patches: should match number of image patches (e.g., 255)

    Returns:
        Tensor of shape [B, target_num_patches]
    """
    B = dcp_batch.shape[0]

    # Try common rectangular layouts
    width = int(round(np.sqrt(target_num_patches)))
    height = int(target_num_patches / width)

    while height * width < target_num_patches:
        width += 1
        height = int(target_num_patches / width)

    target_shape = (width, height)

    resized_maps = []
    for img in dcp_batch:
        # Resize DCP map to match patch layout
        resized = cv2.resize(img, target_shape, interpolation=cv2.INTER_LINEAR)
        resized_maps.append(resized)

    # Flatten spatial dimensions
    dcp_flat = np.array(resized_maps).reshape(B, -1)  # Shape: [B, target_num_patches]

    return torch.tensor(dcp_flat).float()


def dcp_guidance_loss(model_attention, dcp_map):
    B, H, N, _ = model_attention.shape  # e.g., [8, 8, 256, 256]
    target_num_patches = N - 1  # Exclude CLS token → 255

    # Resize DCP maps accordingly
    dcp_flat = resize_dcp_to_patches(dcp_map, target_num_patches=target_num_patches).to(model_attention.device)

    # Extract CLS attention only
    cls_attn = model_attention[:, :, 0, 1:]  # -> [B, H, 255]
    avg_cls_attn = cls_attn.mean(dim=1)  # -> [B, 255]

    # Normalize both distributions
    avg_cls_attn = F.softmax(avg_cls_attn, dim=-1)
    dcp_weights = F.softmax(dcp_flat, dim=-1)

    # Compute KL divergence
    loss = F.kl_div(
        input=torch.log(avg_cls_attn + 1e-8),
        target=dcp_weights,
        reduction='batchmean'
    )

    return loss