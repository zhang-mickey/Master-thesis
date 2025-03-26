import torch
import torch.nn as nn
import torch.nn.functional as F

#Binary segmentation (1 class + background)
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)




#one positive one negative per anchor
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


def pixel_contrastive_loss(features, masks, temperature=0.1,max_samples=256):
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
        H_mask, W_mask = masks.shape[1:]  #512, 512
        feature_map = features[b]  # (C, H, W)
        #If masks were originally 512×512 but feature_map is 128×128, 
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

        #sample a subset of positive and negative features 
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