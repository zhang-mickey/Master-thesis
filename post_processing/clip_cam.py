import torch
import clip
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



class LearnablePrompt(torch.nn.Module):
    def __init__(self, clip_model, num_tokens=5, class_token="smoke"):
        super().__init__()
        self.num_tokens = num_tokens
        self.clip_model = clip_model
        self.ctx = torch.nn.Parameter(torch.randn(num_tokens, clip_model.ln_final.weight.shape[0]))  # [num_tokens, D]
        self.class_token = class_token

    def forward(self):
        # get the embedding of the class token
        with torch.no_grad():
            class_embed = self.clip_model.encode_text(clip.tokenize(self.class_token).to(self.ctx.device))  # [1, D]
        # concatenate
        return torch.cat([self.ctx, class_embed], dim=0)  # [num_tokens+1, D]

# ========== Patch token extraction ==========
@torch.no_grad()
def get_patch_tokens(model, image_tensor):
    # Stem conv
    x = model.visual.conv1(image_tensor)  # [B, C, H, W]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, H*W]
    x = x.permute(0, 2, 1)  # [B, HW, C]

    # Add class token
    class_embed = model.visual.class_embedding.to(x.dtype)
    class_embed = class_embed + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([class_embed, x], dim=1)

    # Positional embedding
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # [HW+1, B, C]
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # [B, HW+1, C]
    # remove CLS token
    return x[:, 1:, :]

def load_image(path, resize_size=224):
    image = Image.open(path).convert("RGB")
    preprocess_custom = T.Compose([
        T.Resize((resize_size, resize_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    return image, preprocess_custom(image).unsqueeze(0).to(device)

# ========== Generate CLIP CAM ==========
@torch.no_grad()
def generate_clip_cam(model, image_tensor, prompts, reshape_size=(7, 7)):
    patch_tokens = get_patch_tokens(model, image_tensor)  # [B, N, D]
    patch_tokens /= patch_tokens.norm(dim=-1, keepdim=True)

    if hasattr(model.visual, 'proj') and model.visual.proj is not None:
        patch_tokens = patch_tokens @ model.visual.proj  # -> [B, N, 512]
        patch_tokens /= patch_tokens.norm(dim=-1, keepdim=True)

    # Encode text
    text_tokens = clip.tokenize(prompts).to(device)

    text_features = model.encode_text(text_tokens)  # [T, 512]
    text_features /= text_features.norm(dim=-1, keepdim=True)

    ## patch_tokens [B, N, D] text_features: [T, D] T: number of prompts
    sim = patch_tokens @ text_features.T  # [B, N, T]
    sim = sim[0, :, 0]  # use first text prompt
    cam = sim.reshape(*reshape_size).cpu().numpy()
    cam = ((cam - cam.min())
           / (cam.max() - cam.min()))
    return cam

def visualize(image_pil, cam, mask_path=None):
    image_np = np.array(image_pil.resize((224, 224))) / 255.0
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255
    overlay = 0.5 * image_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("CLIP CAM")
    plt.imshow(cam_resized, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Overlay")
    plt.imshow(overlay, cmap='jet')
    plt.axis('off')

    if mask_path:
        gt_mask = Image.open(mask_path).convert("L").resize((224, 224))
        gt_np = np.array(gt_mask) / 255.0
        plt.subplot(1, 4, 4)
        plt.title("Ground Truth")
        plt.imshow(gt_np, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "smoke-segmentation.v5i.coco-segmentation/cropped_images/kooks_1__2024-04-14T16-27-29Z_frame_1405_jpg.rf.6ad77a02e4d9070083046a7d106d36dd_896_384.png"
    mask_path = "smoke-segmentation.v5i.coco-segmentation/cropped_masks/mask_kooks_1__2024-04-14T16-27-29Z_frame_1405_jpg.rf.6ad77a02e4d9070083046a7d106d36dd_896_384.png"

    # image_path="smoke-segmentation.v5i.coco-segmentation/cropped_images/kooks_1__2024-11-04T10-44-09Z_frame_464_jpg.rf.a6c659b5080db6ced2ba5ccaf373e79a_256_256.png"
    # mask_path="smoke-segmentation.v5i.coco-segmentation/cropped_masks/mask_kooks_1__2024-11-04T10-44-09Z_frame_464_jpg.rf.a6c659b5080db6ced2ba5ccaf373e79a_256_256.png"

    # prompt = ["a photo of smoke"]
    # prompt = ["a photo of pollution smoke from industrial emissions"]
    prompt = ["a photo of grey industrial smoke rising from factory chimneys"]
    image_pil, image_tensor = load_image(image_path, resize_size=224)
    cam = generate_clip_cam(model, image_tensor, prompt, reshape_size=(7, 7))
    visualize(image_pil, cam, mask_path=mask_path)