import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from lib.network.blocks import *

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size, img_size // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))  # Learnable Positional Encoding

    def forward(self, x):
        B,C,H,W=x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        # x += self.pos_embedding  # Add positional encoding
        return x

class ViTEncoder(nn.Module):
    """
        Args:
            n_heads:Number of attention heads in multi-head self-attention
        Returns:
    """
    def __init__(self, 
                    img_size=512, 
                    patch_size=16, 
                    model_name='vit_b_16', 
                    dropout=0.1,
                    n_layers=12,
                    num_heads=12,
                    num_classes=1,
                    mlp_dim=3072,
                    hidden_dim=768,
                    dim_model=512,
                    drop_path_rate=0.0,
                    pretrained=True,
                    distilled=False):
        super().__init__()

        # self.vit = models.vision_transformer.vit_b_16(pretrained=pretrained)
        # self.vit.heads = nn.Identity()  # Remove classification head
        self.patch_size = patch_size
        self.dropout = nn.Dropout(dropout)
        self.n_layers=n_layers
        self.mlp_dim=mlp_dim
        self.num_heads=num_heads
        self.num_patches = (img_size // patch_size) ** 2  # 1024 patches for 512x512
        self.hidden_dim = hidden_dim  # 768 for ViT-B/16
        self.distilled = distilled

        self.patch_embed=PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=self.hidden_dim
        )
        self.cls_token=nn.Parameter(torch.zeros(1,1,self.hidden_dim))
        #class and position token
        if self.distilled:
            self.dis_token=nn.Parameter(torch.zeros(1,1,self.hidden_dim)) 
            self.pos_embed=nn.Parameter(
                torch.randn(1,self.patch_embed.num_patches+2,self.hidden_dim)
            )
        else:
            self.pos_embed=nn.Parameter(
                torch.randn(1,self.patch_embed.num_patches+1,self.hidden_dim)
            ) # Remove Distillation Head

        # transformer blocks
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,n_layers)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.hidden_dim,
                heads=self.num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(n_layers)
        ])

        #output head
        self.norm=nn.LayerNorm(self.hidden_dim)
        self.head=nn.Linear(self.hidden_dim,num_classes)

        trunc_normal_(self.pos_embed,std=.02)
        trunc_normal_(self.cls_token,std=.02)

        if self.distilled:
            trunc_normal_(self.dis_token,std=.02)

        self.pre_logits=nn.Identity()
        self.apply(init_weights)

    def forward(self, x,return_features=False):
        B, C, H, W = x.shape  # (B, 3, 512, 512)
        PS =self.patch_size
        x = self.patch_embed(x)
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        if self.distilled:
            dis_token = self.dis_token.expand(B, -1, -1)  # (B, 1, dim)
            x = torch.cat((cls_token, dis_token, x), dim=1)  # (B, num_patches + 2, dim)
        else:
            x = torch.cat((cls_token, x), dim=1)  # (B, num_patches + 1, dim)
        # Add positional embeddings
        pos_embed = self.pos_embed
        num_extra_tokens = 1 if not self.distilled else 2
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed=resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H//PS, W//PS),
                num_extra_tokens
            )

        x=x+pos_embed
        x=self.dropout(x)
        for blk in self.blocks:
            x=blk(x)
        x=self.norm(x)
        if return_features:
            return x
        
        if self.distilled:
            x,x_dist=x[:,0],x[:,1]
            x=self.head(x)
        else:
            x=x[:,0]
            x=self.head(x)
        return x
            

# Linear Decoder
class LinearDecoder(nn.Module):
    def __init__(self, num_classes,patch_size,d_encoder):
        super().__init__()
        self.input_dim=d_encoder
        self.output_dim=num_classes
        self.patch_size=patch_size
        self.head=nn.Linear(self.input_dim,self.output_dim)
        self.apply(init_weights)

    def forward(self, x, im_size):
        H, W = im_size
        GS=H//self.patch_size
        x = self.head(x)
        x=rearrange(x,"b (h w) c -> b c h w",h=GS)

        return x

class MaskTransformerDecoder(nn.Module):
    def __init__(self, 
    #hidden size or embedding dimension
                dim_model=512, 
                num_classes=1, 
                num_layers=12, 
                patch_size=16,
                num_heads=8,
                dropout=0.1,
                drop_path_rate=0.0,
                d_encoder=768,
                mlp_dim=3072,
                ):
        super().__init__()
        self.dim_model=dim_model
        self.patch_size=patch_size
       
        self.num_classes=num_classes
        self.scale=dim_model**-0.5

        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,num_layers)]
        
        self.blocks=nn.ModuleList([
            Block(
                dim=dim_model,
                heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])
        self.cls_emb=nn.Parameter(torch.randn(1,num_classes,dim_model))
        self.proj_dec=nn.Linear(d_encoder,dim_model)

        self.proj_patch=nn.Parameter(self.scale*torch.randn(dim_model,dim_model))
        self.proj_classes=nn.Parameter(self.scale*torch.randn(dim_model,dim_model))

        self.decoder_norm=nn.LayerNorm(dim_model)
        self.mask_norm=nn.LayerNorm(num_classes)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb,std=.02)


    def forward(self, x,im_size):
        H,W=im_size
        GS=H//self.patch_size
        # print(f"x shape before proj_dec: {x.shape}")
        #([8, 1024, 768])
        x=self.proj_dec(x)

        B, N, D = x.shape  
        cls_emb=self.cls_emb.expand(B,-1,-1)
        #Before passing into Block, a cls_emb token is added
        x=torch.cat((x,cls_emb),dim=1)

        for blk in self.blocks:
            x=blk(x)

        x=self.decoder_norm(x)
        patches,cls_seg_feat=x[:,:-self.num_classes],x[:,-self.num_classes:]
        patches=patches @ self.proj_patch
        cls_seg_feat=cls_seg_feat @ self.proj_classes

        patches=patches/patches.norm(dim=-1,keepdim=True)
        cls_seg_feat=cls_seg_feat/cls_seg_feat.norm(dim=-1,keepdim=True)

        masks=patches @ cls_seg_feat.transpose(1,2)
        masks=self.mask_norm(masks)
        masks=rearrange(masks,"b (h w) n -> b n h w",h=int(GS))
        return masks

# Segmenter Model
class Segmenter(nn.Module):
    def __init__(self, num_classes, image_size=512, patch_size=16):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = MaskTransformerDecoder()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.image_size = image_size

    def forward(self, x):
        B, C, H_original, W_original = x.shape
        x=padding(x,self.patch_size)
        H,W=x.shape[2:]

        enc_out = self.encoder(x,return_features=True)
        num_extra_tokens=1
        #remove CLS foe decoding
        x=enc_out[:,num_extra_tokens:]  
        # (B, num_patches, 768)
        masks = self.decoder(x,(H,W))

        # print("Decoder Output Shape:", masks.shape)
        masks=F.interpolate(masks,size=(H,W),mode="bilinear")
        masks=unpadding(masks,(H_original,W_original))
        # print("Unpadded Masks Shape:", masks.shape)

        return masks
        
# #       Segmenter model
# #	•	Uses ViT as feature extractor, removing the classification head.
# #	•	Uses patch embedding layer to tokenize the image into patch embeddings.
# #	•	Linear mask transformer decoder (projects tokens into segmentation logits).
# #       employs learnable class embeddings
# #	•	Uses a ViT encoder that extracts features from the image.
# #	•	Uses a Transformer decoder that refines segmentation masks.
# #	•	A Segmenter that combines both and outputs a segmentation map.
# #       Class tokens attend to patch embeddings via cross-attention layers

