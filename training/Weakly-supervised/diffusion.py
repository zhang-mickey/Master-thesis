import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# 配置参数
class RefinementConfig:
    # 输入输出设置
    input_image_path = "smoke-segmentation.v5i.coco-segmentation/cropped_images/kooks_2__2024-11-16T08-49-04Z_frame_1779_jpg.rf.2912d2fed38029058716d204fe6598a8_768_128.png"  # 原始图像路径
    pseudo_label_path = "final_model/transformer_GradCAM_0.3_10_pseudo_labels/fusion_cam_kooks_2__2024-11-16T08-49-04Z_frame_1779_jpg.rf.2912d2fed38029058716d204fe6598a8_768_128.npy"
    # 初始伪标签路径 (.npy))
    output_dir = "diffusion_results"  # 输出目录

    # 扩散模型设置
    model_name = "runwayml/stable-diffusion-v1-5"
    controlnet_name = "lllyasviel/sd-controlnet-seg"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device = "cpu"  # 强制使用CPU
    torch_dtype = torch.float32  # CPU必须使用float32

    # 优化参数
    num_inference_steps = 35  # 扩散迭代步数
    guidance_scale = 8.0  # 文本引导强度
    controlnet_scale = 1.25  # 控制网络强度
    target_size = 512  # 处理尺寸
    batch_size = 1
    seed = 42
    num_iterations = 3  # 迭代优化次数


class ConditionalDiffusionRefiner:
    def __init__(self, config):
        self.config = config
        self._init_models()
        self._prepare_dirs()

    def _init_models(self):
        """初始化扩散模型和控制网络"""
        # 加载ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_name,
            torch_dtype=self.config.torch_dtype
        ).to(self.config.device)

        # 加载扩散模型
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model_name,
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.config.torch_dtype
        ).to(self.config.device)

        # 配置调度器
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        if self.config.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()

    def _prepare_dirs(self):
        """创建输出目录"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_inputs(self):
        """加载图像和伪标签"""
        # 加载原始图像
        self.original_image = Image.open(self.config.input_image_path).convert("RGB")
        self.orig_size = self.original_image.size  # (width, height)

        # 加载伪标签
        self.pseudo_mask = np.load(self.config.pseudo_label_path)

        # 调整伪标签范围到[0,1]
        if self.pseudo_mask.max() > 1 or self.pseudo_mask.min() < 0:
            self.pseudo_mask = (self.pseudo_mask - self.pseudo_mask.min()) / \
                               (self.pseudo_mask.max() - self.pseudo_mask.min())

        print(f"Loaded image: {self.orig_size}, mask range: {self.pseudo_mask.min():.2f}-{self.pseudo_mask.max():.2f}")

    def create_control_image(self, pseudo_mask):
        """创建三通道控制图像"""
        # 通道1: 原始图像灰度化
        gray_img = np.array(self.original_image.convert("L"))
        gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())

        # 通道2: 伪标签（原始）
        mask_img = pseudo_mask

        # 通道3: 边界检测图
        edges = cv2.Canny((gray_img * 255).astype(np.uint8), 50, 150)
        edge_img = edges / 255.0

        # 合并三个通道
        control_image = np.stack([gray_img, mask_img, edge_img], axis=-1)
        return Image.fromarray((control_image * 255).astype(np.uint8))

    def preprocess(self, image, mask, size=None):
        """调整大小预处理"""
        size = size or self.config.target_size
        resized_image = image.resize((size, size), Image.BILINEAR)
        resized_mask = np.array(Image.fromarray(mask).resize((size, size), Image.NEAREST))
        return resized_image, resized_mask

    def postprocess(self, mask, target_size=None):
        """后处理输出掩码"""
        target_size = target_size or self.orig_size
        # 转换为PIL图像并调整大小
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        resized_mask = np.array(mask_pil.resize(target_size, Image.NEAREST)) / 255.0

        # 应用简单阈值处理
        refined_mask = np.where(resized_mask > 0.5, 1.0, 0.0)
        return refined_mask

    def refine_mask(self, prompt=None, negative_prompt=None):
        """执行伪标签优化"""
        generator = torch.Generator(self.config.device).manual_seed(self.config.seed)

        # 准备输入
        resized_image, resized_mask = self.preprocess(
            self.original_image, self.pseudo_mask
        )
        control_image = self.create_control_image(resized_mask)

        # 默认提示词
        prompt = prompt or "clean, precise segmentation mask with accurate boundaries"
        negative_prompt = negative_prompt or "blurry, artifacts, incorrect boundaries"

        # 扩散模型推理
        refined_output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized_image,
            control_image=control_image,
            width=self.config.target_size,
            height=self.config.target_size,
            guidance_scale=self.config.guidance_scale,
            controlnet_conditioning_scale=self.config.controlnet_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=generator,
            num_images_per_prompt=self.config.batch_size,
        ).images[0]

        # 提取掩码（红色通道）
        refined_array = np.array(refined_output)[:, :, 0]
        refined_mask = refined_array / 255.0

        return refined_mask

    def iterative_refinement(self):
        """迭代优化伪标签"""
        current_mask = self.pseudo_mask.copy()

        for i in tqdm(range(self.config.num_iterations), desc="Refining mask"):
            # 临时更新当前掩码
            self.pseudo_mask = current_mask

            # 执行一轮优化
            refined = self.refine_mask()
            refined_resized = self.postprocess(refined)

            # 保存中间结果
            np.save(os.path.join(self.config.output_dir, f"refined_iter_{i}.npy"), refined_resized)

            # 为下一轮准备 - 混合当前结果和优化结果
            current_mask = 0.7 * refined_resized + 0.3 * current_mask
            current_mask = np.clip(current_mask, 0, 1)

        # 最终后处理
        final_mask = np.where(current_mask > 0.5, 1.0, 0.0)
        return final_mask

    def visualize_results(self, refined_mask):
        """可视化结果并保存"""
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        ax[0].imshow(self.original_image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # 初始伪标签
        ax[1].imshow(self.pseudo_mask, cmap="viridis")
        ax[1].set_title("Initial Pseudo Mask")
        ax[1].axis("off")

        # 优化后伪标签
        ax[2].imshow(refined_mask, cmap="viridis")
        ax[2].set_title("Refined Pseudo Mask")
        ax[2].axis("off")

        plt.savefig(os.path.join(self.config.output_dir, "comparison.jpg"), bbox_inches="tight")
        plt.close()

        # 高质量保存优化后掩码
        plt.imsave(os.path.join(self.config.output_dir, "refined_mask.jpg"), refined_mask, cmap="viridis")

    def run(self):
        """执行完整流程"""
        # 1. 加载数据
        self.load_inputs()

        # 2. 迭代优化
        print("Starting refinement process...")
        final_refined_mask = self.iterative_refinement()

        # 3. 保存结果
        np.save(os.path.join(self.config.output_dir, "refined_mask.npy"), final_refined_mask)

        # 4. 可视化
        self.visualize_results(final_refined_mask)

        print(f"Refinement complete! Results saved to: {self.config.output_dir}")

        return final_refined_mask


# 执行优化流程
if __name__ == "__main__":
    config = RefinementConfig()
    refiner = ConditionalDiffusionRefiner(config)
    refined_mask = refiner.run()