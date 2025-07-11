"""
注意力图谱提取与语义掩码生成模块

使用钩子机制从Stable Diffusion的U-Net中提取交叉注意力图谱，
聚合处理后使用Otsu方法生成二值语义掩码。

作者：SAAH-WM团队
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from diffusers import StableDiffusionPipeline

from ..utils.logger_config import LoggerMixin
from ..utils.common_utils import tensor_to_numpy, numpy_to_tensor


class AttentionStore:
    """
    注意力图谱存储器
    
    用于存储从U-Net交叉注意力层捕获的注意力图谱。
    """
    
    def __init__(self):
        """初始化注意力存储器"""
        self.attention_maps = []
        self.step_count = 0
    
    def __call__(self, module, input_tensors, output_tensors):
        """
        钩子函数，在前向传播时被调用
        
        Args:
            module: 被钩住的模块
            input_tensors: 输入张量
            output_tensors: 输出张量
        """
        # 对于交叉注意力，output_tensors通常是一个元组
        # 第一个元素是注意力概率矩阵
        if isinstance(output_tensors, tuple) and len(output_tensors) > 0:
            attention_probs = output_tensors[0]
            
            # 确保是注意力概率矩阵（通常形状为[batch, heads, seq_len, seq_len]）
            if attention_probs is not None and attention_probs.dim() >= 3:
                # 移动到CPU并分离梯度以节省内存
                self.attention_maps.append(attention_probs.detach().cpu())
    
    def reset(self):
        """重置存储器，清除所有注意力图谱"""
        self.attention_maps.clear()
        self.step_count = 0
    
    def get_maps_count(self) -> int:
        """获取存储的注意力图谱数量"""
        return len(self.attention_maps)


class AttentionExtractor(LoggerMixin):
    """
    注意力图谱提取器
    
    管理整个注意力提取流程，包括钩子注册、图谱聚合和掩码生成。
    """
    
    def __init__(self, device: str = "cpu"):
        """
        初始化注意力提取器
        
        Args:
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.attention_store = AttentionStore()
        self.hooks = []  # 存储注册的钩子
        
        self.log_info("注意力图谱提取器初始化完成")
    
    def register_attention_hooks(self, unet):
        """
        为U-Net的交叉注意力层注册钩子
        
        Args:
            unet: Stable Diffusion的U-Net模型
        """
        self.log_info("开始注册注意力钩子...")
        
        # 清除之前的钩子
        self.clear_hooks()
        
        hook_count = 0
        for name, module in unet.named_modules():
            # 查找交叉注意力模块
            # 在diffusers中，交叉注意力通常包含"CrossAttention"或"cross_attention"
            if any(keyword in module.__class__.__name__ for keyword in 
                   ["CrossAttention", "cross_attention", "CrossAttn"]):
                
                hook = module.register_forward_hook(self.attention_store)
                self.hooks.append(hook)
                hook_count += 1
                
                self.log_debug(f"为模块 {name} 注册钩子")
        
        self.log_info(f"成功注册了{hook_count}个注意力钩子")
        
        if hook_count == 0:
            self.log_warning("未找到交叉注意力模块，可能需要调整模块名称匹配规则")
    
    def clear_hooks(self):
        """清除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.log_debug("已清除所有注意力钩子")
    
    def extract_attention_maps(
        self, 
        pipeline: StableDiffusionPipeline,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> List[torch.Tensor]:
        """
        提取注意力图谱
        
        Args:
            pipeline: Stable Diffusion管道
            prompt: 输入prompt
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            
        Returns:
            注意力图谱列表
        """
        self.log_info(f"开始提取注意力图谱，prompt: '{prompt}'")
        
        try:
            # 重置注意力存储器
            self.attention_store.reset()
            
            # 注册钩子
            self.register_attention_hooks(pipeline.unet)
            
            # 生成图像（这会触发钩子收集注意力图谱）
            self.log_info("正在运行Stable Diffusion生成...")
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    return_dict=True
                )
            
            # 获取收集到的注意力图谱
            attention_maps = self.attention_store.attention_maps.copy()
            
            self.log_info(f"成功提取了{len(attention_maps)}个注意力图谱")
            
            return attention_maps, result.images[0]
            
        except Exception as e:
            self.log_error(f"注意力图谱提取失败: {str(e)}")
            raise
        finally:
            # 清除钩子
            self.clear_hooks()
    
    def get_token_indices(self, prompt: str, tokenizer, target_words: List[str]) -> List[int]:
        """
        获取目标词汇在prompt中的token索引
        
        Args:
            prompt: 输入prompt
            tokenizer: 分词器
            target_words: 目标词汇列表
            
        Returns:
            token索引列表
        """
        self.log_debug(f"获取目标词汇的token索引: {target_words}")
        
        # 对prompt进行分词
        tokens = tokenizer.encode(prompt)
        token_strings = [tokenizer.decode([token]) for token in tokens]
        
        self.log_debug(f"Prompt tokens: {token_strings}")
        
        # 查找目标词汇的索引
        target_indices = []
        for target_word in target_words:
            for i, token_str in enumerate(token_strings):
                if target_word.lower() in token_str.lower():
                    target_indices.append(i)
                    self.log_debug(f"找到目标词汇 '{target_word}' 在索引 {i}")
        
        if not target_indices:
            self.log_warning(f"未找到目标词汇 {target_words} 的token索引")
            # 如果没找到，使用所有非特殊token
            target_indices = list(range(1, len(tokens) - 1))  # 排除[CLS]和[SEP]
        
        return target_indices
    
    def get_aggregated_attention_map(
        self, 
        attention_maps: List[torch.Tensor],
        token_indices: List[int],
        target_resolution: int = 64
    ) -> torch.Tensor:
        """
        聚合注意力图谱并调整到目标分辨率
        
        Args:
            attention_maps: 注意力图谱列表
            token_indices: 目标token索引
            target_resolution: 目标分辨率
            
        Returns:
            聚合后的注意力图谱，形状为[target_resolution, target_resolution]
        """
        self.log_info(f"开始聚合{len(attention_maps)}个注意力图谱")
        self.log_debug(f"目标token索引: {token_indices}")
        
        if not attention_maps:
            raise ValueError("注意力图谱列表为空")
        
        aggregated_maps = []
        
        for i, attention_map in enumerate(attention_maps):
            try:
                # attention_map通常形状为[batch, heads, seq_len, seq_len]
                if attention_map.dim() == 4:
                    batch_size, num_heads, seq_len, _ = attention_map.shape
                    
                    # 选择与目标token相关的注意力
                    # 这里我们关注的是文本token对图像patch的注意力
                    # 通常前面的token是图像patch，后面的是文本token
                    
                    # 假设图像patch数量为seq_len的一部分
                    # 对于64x64的潜在空间，通常有64个patch（8x8）
                    num_image_patches = int(np.sqrt(seq_len - len(token_indices)))
                    
                    if num_image_patches * num_image_patches + len(token_indices) <= seq_len:
                        # 提取图像patch对目标token的注意力
                        relevant_attention = attention_map[:, :, :num_image_patches**2, -len(token_indices):]
                        
                        # 在token维度上平均
                        if len(token_indices) > 0:
                            token_attention = relevant_attention.mean(dim=-1)  # [batch, heads, num_patches]
                        else:
                            token_attention = relevant_attention.mean(dim=-1)
                        
                        # 在头和批次维度上平均
                        patch_attention = token_attention.mean(dim=(0, 1))  # [num_patches]
                        
                        # 重塑为方形
                        attention_2d = patch_attention.reshape(num_image_patches, num_image_patches)
                        
                        aggregated_maps.append(attention_2d)
                        
            except Exception as e:
                self.log_warning(f"处理第{i}个注意力图谱时出错: {str(e)}")
                continue
        
        if not aggregated_maps:
            self.log_error("没有成功处理的注意力图谱")
            # 返回一个默认的注意力图谱
            return torch.ones(target_resolution, target_resolution) * 0.5
        
        # 在所有图谱上平均
        final_attention = torch.stack(aggregated_maps).mean(dim=0)
        
        # 调整到目标分辨率
        if final_attention.shape[0] != target_resolution:
            final_attention = final_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            final_attention = F.interpolate(
                final_attention,
                size=(target_resolution, target_resolution),
                mode='bilinear',
                align_corners=False
            )
            final_attention = final_attention.squeeze()  # [target_resolution, target_resolution]
        
        self.log_info(f"注意力图谱聚合完成，最终形状: {final_attention.shape}")
        return final_attention

    def generate_semantic_mask(self, attention_map: torch.Tensor) -> torch.Tensor:
        """
        使用Otsu方法从注意力图谱生成二值语义掩码

        Args:
            attention_map: 注意力图谱，形状为[H, W]

        Returns:
            二值掩码，形状为[1, 1, H, W]，值为0或1
        """
        self.log_info("开始生成语义掩码...")

        try:
            # 将张量转换为numpy数组
            attention_np = tensor_to_numpy(attention_map)

            # 归一化到0-255范围
            attention_normalized = ((attention_np - attention_np.min()) /
                                  (attention_np.max() - attention_np.min() + 1e-8) * 255)
            attention_uint8 = attention_normalized.astype(np.uint8)

            self.log_debug(f"注意力图谱统计: min={attention_np.min():.4f}, "
                          f"max={attention_np.max():.4f}, mean={attention_np.mean():.4f}")

            # 使用Otsu方法进行自动阈值分割
            threshold_value, binary_mask = cv2.threshold(
                attention_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            self.log_info(f"Otsu阈值: {threshold_value}")

            # 转换回张量格式
            mask_tensor = torch.from_numpy(binary_mask / 255.0).float()

            # 添加批次和通道维度 [H, W] -> [1, 1, H, W]
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

            # 移动到指定设备
            mask_tensor = mask_tensor.to(self.device)

            # 计算前景和背景像素比例
            foreground_ratio = (mask_tensor == 1).float().mean().item()
            background_ratio = 1 - foreground_ratio

            self.log_info(f"语义掩码生成完成，前景比例: {foreground_ratio:.3f}, "
                         f"背景比例: {background_ratio:.3f}")

            return mask_tensor

        except Exception as e:
            self.log_error(f"语义掩码生成失败: {str(e)}")
            raise

    def extract_and_generate_mask(
        self,
        pipeline: StableDiffusionPipeline,
        prompt: str,
        target_words: Optional[List[str]] = None,
        target_resolution: int = 64,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的注意力提取和掩码生成流程

        Args:
            pipeline: Stable Diffusion管道
            prompt: 输入prompt
            target_words: 目标词汇列表，如果为None则自动提取
            target_resolution: 目标分辨率
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度

        Returns:
            (聚合注意力图谱, 二值语义掩码)
        """
        self.log_info("开始完整的注意力提取和掩码生成流程")

        try:
            # 步骤1：提取注意力图谱
            attention_maps, generated_image = self.extract_attention_maps(
                pipeline, prompt, num_inference_steps, guidance_scale
            )

            # 步骤2：获取目标词汇的token索引
            if target_words is None:
                # 简单的词汇提取：去除常见停用词
                stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                words = prompt.lower().split()
                target_words = [word for word in words if word not in stop_words]
                self.log_info(f"自动提取的目标词汇: {target_words}")

            token_indices = self.get_token_indices(prompt, pipeline.tokenizer, target_words)

            # 步骤3：聚合注意力图谱
            aggregated_attention = self.get_aggregated_attention_map(
                attention_maps, token_indices, target_resolution
            )

            # 步骤4：生成二值掩码
            semantic_mask = self.generate_semantic_mask(aggregated_attention)

            self.log_info("完整流程执行成功")
            return aggregated_attention, semantic_mask, generated_image

        except Exception as e:
            self.log_error(f"完整流程执行失败: {str(e)}")
            raise

    def visualize_attention_and_mask(
        self,
        attention_map: torch.Tensor,
        mask: torch.Tensor,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        可视化注意力图谱和掩码

        Args:
            attention_map: 注意力图谱
            mask: 二值掩码
            save_path: 保存路径

        Returns:
            可视化图像的numpy数组
        """
        import matplotlib.pyplot as plt

        # 转换为numpy数组
        attention_np = tensor_to_numpy(attention_map)
        mask_np = tensor_to_numpy(mask.squeeze())

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始注意力图谱
        im1 = axes[0].imshow(attention_np, cmap='hot', interpolation='nearest')
        axes[0].set_title('原始注意力图谱')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # 二值掩码
        im2 = axes[1].imshow(mask_np, cmap='gray', interpolation='nearest')
        axes[1].set_title('二值语义掩码')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # 叠加显示
        overlay = attention_np.copy()
        overlay[mask_np == 1] = overlay[mask_np == 1] * 0.7 + 0.3  # 高亮前景区域
        im3 = axes[2].imshow(overlay, cmap='hot', interpolation='nearest')
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log_info(f"可视化结果已保存到: {save_path}")

        # 转换为numpy数组返回
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return vis_array

    def get_extraction_stats(self) -> dict:
        """
        获取提取统计信息

        Returns:
            统计信息字典
        """
        return {
            "attention_maps_count": self.attention_store.get_maps_count(),
            "hooks_registered": len(self.hooks),
            "device": self.device
        }
