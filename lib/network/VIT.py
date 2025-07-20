# ---------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/rwightman/pytorch-image-models
# ---------------------------------------------------------------------------------------------------------------
""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
import torch.utils.model_zoo as model_zoo
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import resnet26d, resnet50d
from timm.models.registry import register_model
import os


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1, 'input_size': (3, 512, 512), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


def resize_pos_embed(pos_embed_checkpoint, pos_embed_model):
    """
    Rescales position embeddings from the checkpoint to fit the model.
    """
    num_extra_tokens = 1  # For class token
    embedding_dim = pos_embed_model.shape[-1]  # Should be 768

    # Extract CLS token embedding
    cls_token_embed = pos_embed_checkpoint[:, :num_extra_tokens, :]
    pos_embed_checkpoint = pos_embed_checkpoint[:, num_extra_tokens:, :]

    # Compute number of patches
    num_patches_checkpoint = pos_embed_checkpoint.shape[1]
    num_patches_model = pos_embed_model.shape[1] - num_extra_tokens  # Exclude CLS token

    # Compute grid size
    grid_size_checkpoint = int(num_patches_checkpoint ** 0.5)
    grid_size_model = int(num_patches_model ** 0.5)

    # Reshape to grid format
    pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, grid_size_checkpoint, grid_size_checkpoint, embedding_dim)
    pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 3, 1, 2)  # Shape: (1, 768, grid, grid)

    # Resize using bilinear interpolation
    pos_embed_checkpoint = F.interpolate(pos_embed_checkpoint, size=(grid_size_model, grid_size_model), mode='bilinear',
                                         align_corners=False)

    # Reshape back
    pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 3, 1).reshape(1, num_patches_model, embedding_dim)

    # Concatenate CLS token back
    return torch.cat([cls_token_embed, pos_embed_checkpoint], dim=1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # weights = attn if self.vis else None
        weights = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False,
#                  drop_token_ratio=0.1):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim
#         self.drop_token_ratio = drop_token_ratio

#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
#         else:
#             self.patch_embed = PatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

#         num_patches = self.patch_embed.num_patches
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         self._size = img_size // patch_size

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def prepare_tokens(self, x):
#         B, nc, h, w = x.shape
#         h, w = h // self.patch_embed.patch_size[0], w // self.patch_embed.patch_size[1]
#         x = self.patch_embed(x)

#         patch_pos_embed = self.pos_embed[:, 1:, :].reshape(1, self._size, self._size, -1).permute(0, 3, 1, 2)
#         patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode="bicubic", align_corners=False)
#         patch_pos_embed = patch_pos_embed.reshape(1, -1, h * w).permute(0, 2, 1)
#         pos_embed = torch.cat((self.pos_embed[:, :1, :], patch_pos_embed), dim=1)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + pos_embed
#         return x

#     def apply_token_dropout(self, x):
#         if not self.training or self.drop_token_ratio == 0.0:
#             return x

#         cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]
#         B, N, D = patch_tokens.shape
#         drop_num = int(N * self.drop_token_ratio)

#         mask = torch.ones((B, N), dtype=torch.bool, device=x.device)
#         for i in range(B):
#             drop_idx = torch.randperm(N, device=x.device)[:drop_num]
#             mask[i, drop_idx] = False

#         patch_tokens[~mask] = 0  # zero out dropped tokens
#         return torch.cat([cls_token, patch_tokens], dim=1)

#     def forward_features(self, x):
#         x = self.prepare_tokens(x)
#         x = self.pos_drop(x)
#         x = self.apply_token_dropout(x)

#         attn_weights = []
#         cls_embeddings = []
#         for blk in self.blocks:
#             x, weights = blk(x)
#             attn_weights.append(weights)
#             cls_embeddings.append(x[:, 0])

#         x = self.norm(x)
#         cls_embeddings.append(x[:, 0])
#         return x[:, 0], attn_weights, cls_embeddings

#     def forward(self, x):
#         x, attn_weights, _ = self.forward_features(x)
#         x = self.head(x)
#         return x, attn_weights

# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False,
#                  drop_token_ratio=0.1):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim
#         self.drop_token_ratio = drop_token_ratio

#         # 添加可学习的MASK标记
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 新增

#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
#         else:
#             self.patch_embed = PatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

#         num_patches = self.patch_embed.num_patches
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         self._size = img_size // patch_size

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.mask_token, std=.02)  # 新增初始化
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'mask_token'}  # 新增mask_token

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def prepare_tokens(self, x):
#         B, nc, h, w = x.shape
#         h, w = h // self.patch_embed.patch_size[0], w // self.patch_embed.patch_size[1]
#         x = self.patch_embed(x)

#         patch_pos_embed = self.pos_embed[:, 1:, :].reshape(1, self._size, self._size, -1).permute(0, 3, 1, 2)
#         patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode="bicubic", align_corners=False)
#         patch_pos_embed = patch_pos_embed.reshape(1, -1, h * w).permute(0, 2, 1)
#         pos_embed = torch.cat((self.pos_embed[:, :1, :], patch_pos_embed), dim=1)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + pos_embed
#         return x

#     def apply_token_dropout(self, x):
#         if not self.training or self.drop_token_ratio == 0.0:
#             return x

#         cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]
#         B, N, D = patch_tokens.shape
#         drop_num = int(N * self.drop_token_ratio)

#         # 创建丢弃掩码
#         mask = torch.ones((B, N), dtype=torch.bool, device=x.device)
#         for i in range(B):
#             drop_idx = torch.randperm(N, device=x.device)[:drop_num]
#             mask[i, drop_idx] = False

#         # 使用MASK标记替换被丢弃的token (改进点)
#         mask_tokens = self.mask_token.expand(B, drop_num, D)  # 扩展MASK标记
#         patch_tokens = patch_tokens.clone()  # 确保不修改原始tensor
#         patch_tokens[~mask] = mask_tokens.reshape(B * drop_num, D)  # 替换

#         return torch.cat([cls_token, patch_tokens], dim=1)

#     def forward_features(self, x):
#         x = self.prepare_tokens(x)
#         x = self.pos_drop(x)
#         x = self.apply_token_dropout(x)  # 应用改进的token dropout

#         attn_weights = []
#         cls_embeddings = []
#         for blk in self.blocks:
#             x, weights = blk(x)
#             attn_weights.append(weights)
#             cls_embeddings.append(x[:, 0])

#         x = self.norm(x)
#         cls_embeddings.append(x[:, 0])
#         return x[:, 0], attn_weights, cls_embeddings

#     def forward(self, x):
#         x, attn_weights, _ = self.forward_features(x)
#         x = self.head(x)
#         return x, attn_weights


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        # It has the same dimensionality as the patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.feature_shape = None
        self._size = img_size // patch_size
        self.gradients = None
        #the features are mapped into the same dimension by a linear projection
        self.vit_proj = nn.Conv2d(embed_dim, 2, kernel_size=1) 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        h, w = h // self.patch_embed.patch_size[0], w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)  # patch linear embedding

        patch_pos_embed = self.pos_embed[:, 1:, :].reshape(1, self._size, self._size, -1).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode="bicubic", align_corners=False)
        patch_pos_embed = patch_pos_embed.reshape(1, -1, h * w).permute(0, 2, 1)
        pos_embed = torch.cat((self.pos_embed[:, :1, :], patch_pos_embed), dim=1)
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        x = x + pos_embed
        return x

    def forward_features(self, x):
        # B = x.shape[0]
        # x = self.patch_embed(x)
        x = self.prepare_tokens(x)
        x = self.pos_drop(x)
        attn_weights = []
        cls_embeddings = []  # Store CLS tokens from all layers
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
            cls_embeddings.append(x[:, 0])
        x = self.norm(x)
        cls_embeddings.append(x[:, 0])
        if self.feature_shape is None:
            self.feature_shape = (self.embed_dim, self._size, self._size)
        # 返回所有patch tokens，不包括CLS token
        patch_tokens = x[:, 1:]
        B, num_patches, embed_dim = patch_tokens.shape
        H = W = int(num_patches ** 0.5)
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, embed_dim, H, W)
        return x[:, 0], attn_weights, cls_embeddings, feature_map

    def forward(self, x):
        x, attn_weights, _, feature_map = self.forward_features(x)
        x = self.head(x)
        feature_map_2ch = self.vit_proj(feature_map)
        return x, feature_map, feature_map_2ch


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, img_size=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                       "pretrained", "vit_small_p16_224-15ec54c9.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "pos_embed" in checkpoint:
            checkpoint["pos_embed"] = resize_pos_embed(checkpoint["pos_embed"], model.pos_embed)
        # Remove the classifier head from the checkpoint
        checkpoint.pop("head.weight", None)
        checkpoint.pop("head.bias", None)

        model.load_state_dict(checkpoint, strict=False)

        # Reinitialize the classification head to match num_classes
        model.head = nn.Linear(768, num_classes)
    return model


@register_model
def backbone_vit_base_patch16_224(pretrained=True, num_classes=1, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, img_size=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = default_cfgs['vit_base_patch16_224']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                       "pretrained", "jx_vit_base_p16_224-80ecf9dd.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "pos_embed" in checkpoint:
            checkpoint["pos_embed"] = resize_pos_embed(checkpoint["pos_embed"], model.pos_embed)
        # Remove the classifier head from the checkpoint
        checkpoint.pop("head.weight", None)
        checkpoint.pop("head.bias", None)

        model.load_state_dict(checkpoint, strict=False)

        # Reinitialize the classification head to match num_classes
        model.head = nn.Linear(768, num_classes)
    return model


def compute_gradcam(activations, grads):
    # 平均每个通道上的梯度：全局平均池化
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * activations).sum(dim=1)  # [B, H, W]
    cam = F.relu(cam)  # 只保留正值
    cam = cam - cam.min(dim=(1, 2), keepdim=True)[0]
    cam = cam / (cam.max(dim=(1, 2), keepdim=True)[0] + 1e-8)
    return cam


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model


@register_model
def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet26d_224']
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet50d_s3_224']
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet26d_224']
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet50d_224']
    return model