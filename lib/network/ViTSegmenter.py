import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import trunc_normal_

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))  # Learnable Positional Encoding

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x += self.pos_embedding  # Add positional encoding
        return x

# ViT Encoder (Removes Classification Head)
class ViTEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16, model_name='vit_b_16', pretrained=True):
        super().__init__()

        self.vit = models.vision_transformer.vit_b_16(pretrained=pretrained)
        self.vit.heads = nn.Identity()  # Remove classification head

        self.num_patches = (img_size // patch_size) ** 2  # 1024 patches for 512x512
        self.hidden_dim = self.vit.hidden_dim  # 768 for ViT-B/16

        # Get original positional embedding (1, 197, 768)
        orig_pos_embed = self.vit.encoder.pos_embedding  
        cls_pos_embed = orig_pos_embed[:, :1, :]  # Keep class token pos embedding
        patch_pos_embed = orig_pos_embed[:, 1:, :]  # Patch embeddings (196 patches)

        # Reshape & interpolate patches to match the new num_patches (1024)
        patch_pos_embed = patch_pos_embed.reshape(1, 14, 14, self.hidden_dim).permute(0, 3, 1, 2)  # (1, 768, 14, 14)
        new_patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(img_size // patch_size, img_size // patch_size),  # (32, 32) for 512x512
            mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1).reshape(1, self.num_patches, self.hidden_dim)  # (1, 1024, 768)

        # Concatenate class token embedding with resized patch embeddings
        self.vit.encoder.pos_embedding = nn.Parameter(torch.cat([cls_pos_embed, new_patch_pos_embed], dim=1))  # (1, 1025, 768)

    def forward(self, x):
        B, C, H, W = x.shape  # (B, 3, 512, 512)
        x = self.vit.conv_proj(x)  # (B, dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        # Add class token
        cls_token = self.vit.class_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches + 1, dim)

        # Pass through ViT encoder
        x = self.vit.encoder(x)  # (B, num_patches + 1, dim)

        return x[:, 1:, :]  # Remove class token before returning

# Linear Decoder
class LinearDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = nn.Linear(in_channels, out_channels)

    def forward(self, x, im_size):
        H, W = im_size
        x = self.head(x)
        return x

# Mask Transformer Decoder with Cross-Attention
class MaskTransformerDecoder(nn.Module):
    def __init__(self, dim=768, num_classes=1, num_layers=3, num_heads=8):
        super().__init__()
        self.class_embeds = nn.Parameter(torch.randn(num_classes, dim))  # Learnable class tokens
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=2048)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # self.segmentation_head = nn.Linear(dim, num_classes)  # Predict segmentation masks

    def forward(self, x):
        """
        Args:
            patch_embeds: (B, num_patches, dim)  # e.g., (8, 1024, 768)
        Returns:
            masks: (B, num_classes, num_patches)  # e.g., (8, 1, 1024)
        """
        B, N, D = x.shape  
        # (Batch, Patches, Embedding Dim)
        # Class tokens: (num_classes, dim) → (B, num_classes, dim)
        class_tokens = self.class_embeds.unsqueeze(1).expand(B, -1, -1)
        
        # Cross-attention: class tokens (query) → patch_embeds (key/value)
        # Input shapes for PyTorch Transformer:
        # - tgt: (sequence_length, B, dim) → (num_classes, B, dim)
        # - memory: (sequence_length, B, dim) → (1024, B, dim)
        class_features = self.transformer_decoder(
            class_tokens.permute(1, 0, 2),  # (num_classes, B, dim)
            x.permute(1, 0, 2)   # (1024, B, dim)
        ).permute(1, 0, 2)  # (B, num_classes, dim)

        
        masks = torch.einsum('bcd,bpd->bcp', class_features, x)
        return masks

# Segmenter Model
class Segmenter(nn.Module):
    def __init__(self, num_classes, image_size=512, patch_size=16):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = MaskTransformerDecoder(dim=768, num_classes=num_classes)
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, x):
        B, C, H, W = x.shape
        enc_out = self.encoder(x)  # (B, num_patches, 768)
        dec_out = self.decoder(enc_out)  # (B, num_classes, num_patches)
        print(f"Decoder output shape: {dec_out.shape}")

        grid_size = H // self.patch_size
        print(f"Computed grid_size: {grid_size}")
        dec_out = dec_out.view(B, -1, grid_size, grid_size)  # Reshape to spatial
        dec_out = F.interpolate(dec_out, size=(H, W), mode='bilinear', align_corners=False)  # Upsample

        return dec_out
        
# #       Segmenter model
# #	•	Uses ViT as feature extractor, removing the classification head.
# #	•	Uses patch embedding layer to tokenize the image into patch embeddings.
# #	•	Linear mask transformer decoder (projects tokens into segmentation logits).
# #       employs learnable class embeddings
# #	•	Uses a ViT encoder that extracts features from the image.
# #	•	Uses a Transformer decoder that refines segmentation masks.
# #	•	A Segmenter that combines both and outputs a segmentation map.
# #       Class tokens attend to patch embeddings via cross-attention layers

