"""
基准水印生成模块

使用语义哈希作为种子，通过确定性伪随机数生成器生成基准水印。
确保相同的语义哈希总是生成相同的水印模式。

作者：SAAH-WM团队
"""

import torch
import numpy as np
from typing import Tuple, Optional

from ..utils.logger_config import LoggerMixin
from ..utils.common_utils import bits_to_int, validate_bits


class BaseWatermarkGenerator(LoggerMixin):
    """
    基准水印生成器
    
    使用语义哈希c_bits作为种子，生成确定性的随机水印模式。
    水印在潜在空间中表示，通常为64x64x4的张量。
    """
    
    def __init__(self, device: str = "cpu"):
        """
        初始化基准水印生成器
        
        Args:
            device: 计算设备
        """
        super().__init__()
        self.device = device
        self.log_info(f"基准水印生成器初始化完成，设备: {device}")
    
    def _seed_from_bits(self, c_bits: str) -> int:
        """
        将二进制字符串转换为随机种子
        
        Args:
            c_bits: 二进制字符串
            
        Returns:
            随机种子整数
        """
        if not validate_bits(c_bits):
            raise ValueError(f"输入不是有效的二进制字符串: {c_bits[:32]}...")
        
        # 将二进制字符串转换为整数
        seed = bits_to_int(c_bits)
        
        # 确保种子在合理范围内（Python的random模块要求）
        # 使用模运算限制在32位整数范围内
        seed = seed % (2**32)
        
        self.log_debug(f"从{len(c_bits)}位二进制字符串生成种子: {seed}")
        return seed
    
    def generate_base_watermark(
        self, 
        c_bits: str, 
        height: int = 64, 
        width: int = 64, 
        channels: int = 4
    ) -> torch.Tensor:
        """
        生成基准水印张量
        
        Args:
            c_bits: 语义哈希二进制字符串
            height: 水印高度
            width: 水印宽度  
            channels: 通道数
            
        Returns:
            形状为[1, channels, height, width]的水印张量
        """
        self.log_info(f"开始生成基准水印，尺寸: [1, {channels}, {height}, {width}]")
        self.log_debug(f"使用语义哈希: {c_bits[:32]}...{c_bits[-32:]}")
        
        try:
            # 步骤1：从二进制字符串生成种子
            seed = self._seed_from_bits(c_bits)
            self.log_debug(f"生成的随机种子: {seed}")
            
            # 步骤2：设置PyTorch随机种子
            torch.manual_seed(seed)
            self.log_debug("已设置PyTorch随机种子")
            
            # 步骤3：生成高频随机水印
            # 使用randn生成标准正态分布的随机数，适合作为高频水印
            w_base = torch.randn(
                (1, channels, height, width), 
                dtype=torch.float32,
                device=self.device
            )
            
            self.log_info(f"基准水印生成完成，形状: {w_base.shape}")
            self.log_debug(f"水印统计信息 - 均值: {w_base.mean().item():.6f}, "
                          f"标准差: {w_base.std().item():.6f}, "
                          f"最小值: {w_base.min().item():.6f}, "
                          f"最大值: {w_base.max().item():.6f}")
            
            return w_base
            
        except Exception as e:
            self.log_error(f"基准水印生成失败: {str(e)}")
            raise
    
    def verify_deterministic_generation(
        self, 
        c_bits: str, 
        height: int = 64, 
        width: int = 64, 
        channels: int = 4,
        num_tests: int = 5
    ) -> bool:
        """
        验证相同输入是否产生相同的水印
        
        Args:
            c_bits: 语义哈希二进制字符串
            height: 水印高度
            width: 水印宽度
            channels: 通道数
            num_tests: 测试次数
            
        Returns:
            是否确定性生成
        """
        self.log_info(f"开始验证确定性生成，测试次数: {num_tests}")
        
        watermarks = []
        for i in range(num_tests):
            w_base = self.generate_base_watermark(c_bits, height, width, channels)
            watermarks.append(w_base)
            self.log_debug(f"第{i+1}次测试完成")
        
        # 检查所有水印是否完全相同
        reference = watermarks[0]
        is_deterministic = True
        
        for i, watermark in enumerate(watermarks[1:], 1):
            if not torch.equal(reference, watermark):
                self.log_error(f"第{i+1}次生成的水印与参考水印不同")
                is_deterministic = False
                break
        
        if is_deterministic:
            self.log_info("确定性生成验证通过")
        else:
            self.log_error("确定性生成验证失败")
            
        return is_deterministic
    
    def generate_watermark_batch(
        self, 
        c_bits_list: list, 
        height: int = 64, 
        width: int = 64, 
        channels: int = 4
    ) -> torch.Tensor:
        """
        批量生成基准水印
        
        Args:
            c_bits_list: 语义哈希列表
            height: 水印高度
            width: 水印宽度
            channels: 通道数
            
        Returns:
            形状为[batch_size, channels, height, width]的水印张量
        """
        batch_size = len(c_bits_list)
        self.log_info(f"开始批量生成基准水印，批次大小: {batch_size}")
        
        watermarks = []
        for i, c_bits in enumerate(c_bits_list):
            self.log_progress(i + 1, batch_size, "批量水印生成")
            w_base = self.generate_base_watermark(c_bits, height, width, channels)
            watermarks.append(w_base)
        
        # 拼接为批次张量
        batch_watermarks = torch.cat(watermarks, dim=0)
        
        self.log_info(f"批量水印生成完成，形状: {batch_watermarks.shape}")
        return batch_watermarks
    
    def compare_watermarks(self, w1: torch.Tensor, w2: torch.Tensor) -> dict:
        """
        比较两个水印的相似性
        
        Args:
            w1: 第一个水印张量
            w2: 第二个水印张量
            
        Returns:
            相似性统计信息
        """
        if w1.shape != w2.shape:
            raise ValueError(f"水印形状不匹配: {w1.shape} vs {w2.shape}")
        
        # 计算各种距离度量
        mse = torch.mean((w1 - w2) ** 2).item()
        mae = torch.mean(torch.abs(w1 - w2)).item()
        
        # 计算相关系数
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()
        correlation = torch.corrcoef(torch.stack([w1_flat, w2_flat]))[0, 1].item()
        
        # 计算余弦相似度
        cosine_sim = torch.nn.functional.cosine_similarity(
            w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)
        ).item()
        
        stats = {
            "mse": mse,
            "mae": mae,
            "correlation": correlation,
            "cosine_similarity": cosine_sim,
            "are_equal": torch.equal(w1, w2)
        }
        
        self.log_debug(f"水印比较结果: MSE={mse:.6f}, MAE={mae:.6f}, "
                      f"相关系数={correlation:.6f}, 余弦相似度={cosine_sim:.6f}")
        
        return stats
    
    def get_watermark_statistics(self, watermark: torch.Tensor) -> dict:
        """
        获取水印的统计信息
        
        Args:
            watermark: 水印张量
            
        Returns:
            统计信息字典
        """
        stats = {
            "shape": list(watermark.shape),
            "dtype": str(watermark.dtype),
            "device": str(watermark.device),
            "mean": watermark.mean().item(),
            "std": watermark.std().item(),
            "min": watermark.min().item(),
            "max": watermark.max().item(),
            "num_elements": watermark.numel()
        }
        
        return stats
