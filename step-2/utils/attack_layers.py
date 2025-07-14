"""
SAAH-WM Baseline 第二步 - 攻击模拟层模块

本模块实现各种图像攻击的模拟，用于训练鲁棒水印系统。
攻击类型包括JPEG压缩、高斯噪声、缩放等常见的图像处理操作。

攻击的目的是测试水印的鲁棒性，确保嵌入的信息在经过攻击后仍能被正确解码。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import math
from typing import Dict, List, Tuple, Optional

# 配置日志
logger = logging.getLogger(__name__)


class JPEGCompression(nn.Module):
    """
    JPEG压缩攻击模拟
    
    通过量化操作模拟JPEG压缩对图像的影响。
    
    Args:
        quality_range (Tuple[int, int]): 质量因子范围，默认(50, 95)
        probability (float): 应用攻击的概率，默认0.7
    """
    
    def __init__(self, quality_range: Tuple[int, int] = (50, 95), probability: float = 0.7):
        super(JPEGCompression, self).__init__()
        
        self.quality_range = quality_range
        self.probability = probability
        
        logger.info(f"初始化JPEG压缩攻击: 质量范围={quality_range}, 概率={probability}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用JPEG压缩攻击
        
        Args:
            x (torch.Tensor): 输入图像张量，形状 [B, C, H, W]
            
        Returns:
            torch.Tensor: 攻击后的图像张量
        """
        if random.random() > self.probability:
            return x
            
        batch_size = x.shape[0]
        attacked_images = []
        
        for i in range(batch_size):
            # 随机选择质量因子
            quality = random.randint(self.quality_range[0], self.quality_range[1])
            
            # 简化的JPEG压缩模拟：通过量化实现
            # 实际应用中可以使用更精确的JPEG压缩算法
            image = x[i]
            
            # 量化强度与质量因子成反比
            quantization_factor = 100.0 / quality
            
            # 应用量化
            quantized = torch.round(image * quantization_factor) / quantization_factor
            
            # 添加轻微噪声模拟压缩伪影
            noise_std = (100 - quality) / 1000.0
            noise = torch.randn_like(quantized) * noise_std
            compressed = quantized + noise
            
            attacked_images.append(compressed)
            
        result = torch.stack(attacked_images, dim=0)
        
        logger.debug(f"JPEG压缩攻击完成: 输入形状={x.shape}, 输出形状={result.shape}")
        
        return result


class GaussianNoise(nn.Module):
    """
    高斯噪声攻击
    
    向图像添加高斯噪声，模拟传输或存储过程中的噪声干扰。
    
    Args:
        sigma_range (Tuple[float, float]): 噪声标准差范围，默认(0.0, 0.1)
        probability (float): 应用攻击的概率，默认0.5
    """
    
    def __init__(self, sigma_range: Tuple[float, float] = (0.0, 0.1), probability: float = 0.5):
        super(GaussianNoise, self).__init__()
        
        self.sigma_range = sigma_range
        self.probability = probability
        
        logger.info(f"初始化高斯噪声攻击: 标准差范围={sigma_range}, 概率={probability}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用高斯噪声攻击
        
        Args:
            x (torch.Tensor): 输入图像张量，形状 [B, C, H, W]
            
        Returns:
            torch.Tensor: 攻击后的图像张量
        """
        if random.random() > self.probability:
            return x
            
        # 随机选择噪声标准差
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        # 生成高斯噪声
        noise = torch.randn_like(x) * sigma
        
        # 添加噪声
        noisy_image = x + noise
        
        logger.debug(f"高斯噪声攻击完成: 标准差={sigma:.4f}, 输入形状={x.shape}")
        
        return noisy_image


class ResizeAttack(nn.Module):
    """
    缩放攻击
    
    通过缩放操作改变图像尺寸，然后恢复原尺寸，模拟图像缩放过程中的信息损失。
    
    Args:
        scale_range (Tuple[float, float]): 缩放比例范围，默认(0.8, 1.2)
        probability (float): 应用攻击的概率，默认0.3
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), probability: float = 0.3):
        super(ResizeAttack, self).__init__()
        
        self.scale_range = scale_range
        self.probability = probability
        
        logger.info(f"初始化缩放攻击: 缩放范围={scale_range}, 概率={probability}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用缩放攻击
        
        Args:
            x (torch.Tensor): 输入图像张量，形状 [B, C, H, W]
            
        Returns:
            torch.Tensor: 攻击后的图像张量
        """
        if random.random() > self.probability:
            return x
            
        original_size = x.shape[2:]
        
        # 随机选择缩放比例
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 计算新尺寸
        new_height = int(original_size[0] * scale)
        new_width = int(original_size[1] * scale)
        
        # 缩放到新尺寸
        resized = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)
        
        # 恢复原尺寸
        restored = F.interpolate(resized, size=original_size, mode='bilinear', align_corners=False)
        
        logger.debug(f"缩放攻击完成: 缩放比例={scale:.3f}, 中间尺寸=({new_height}, {new_width})")
        
        return restored


class RotationAttack(nn.Module):
    """
    旋转攻击
    
    对图像进行小角度旋转，然后旋转回来，模拟旋转过程中的信息损失。
    
    Args:
        angle_range (Tuple[float, float]): 旋转角度范围（度），默认(-5, 5)
        probability (float): 应用攻击的概率，默认0.2
    """
    
    def __init__(self, angle_range: Tuple[float, float] = (-5, 5), probability: float = 0.2):
        super(RotationAttack, self).__init__()
        
        self.angle_range = angle_range
        self.probability = probability
        
        logger.info(f"初始化旋转攻击: 角度范围={angle_range}, 概率={probability}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用旋转攻击
        
        Args:
            x (torch.Tensor): 输入图像张量，形状 [B, C, H, W]
            
        Returns:
            torch.Tensor: 攻击后的图像张量
        """
        if random.random() > self.probability:
            return x
            
        # 随机选择旋转角度
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        angle_rad = math.radians(angle)
        
        # 创建旋转矩阵
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 构建仿射变换矩阵
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        # 生成网格
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # 应用旋转
        rotated = F.grid_sample(x, grid, align_corners=False)
        
        # 旋转回来
        theta_inv = torch.tensor([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        
        grid_inv = F.affine_grid(theta_inv, x.size(), align_corners=False)
        restored = F.grid_sample(rotated, grid_inv, align_corners=False)
        
        logger.debug(f"旋转攻击完成: 角度={angle:.2f}度")
        
        return restored


class AttackLayer(nn.Module):
    """
    攻击层：组合多种攻击方法
    
    随机选择并应用一种或多种攻击方法，模拟真实环境中的各种图像处理操作。
    
    Args:
        attack_config (Dict): 攻击配置字典
    """
    
    def __init__(self, attack_config: Dict):
        super(AttackLayer, self).__init__()
        
        self.attack_config = attack_config
        self.attacks = nn.ModuleDict()
        
        # 初始化各种攻击
        if attack_config.get('jpeg_compression', {}).get('enabled', False):
            jpeg_config = attack_config['jpeg_compression']
            self.attacks['jpeg'] = JPEGCompression(
                quality_range=tuple(jpeg_config['quality_range']),
                probability=jpeg_config['probability']
            )
            
        if attack_config.get('gaussian_noise', {}).get('enabled', False):
            noise_config = attack_config['gaussian_noise']
            self.attacks['noise'] = GaussianNoise(
                sigma_range=tuple(noise_config['sigma_range']),
                probability=noise_config['probability']
            )
            
        if attack_config.get('resize_attack', {}).get('enabled', False):
            resize_config = attack_config['resize_attack']
            self.attacks['resize'] = ResizeAttack(
                scale_range=tuple(resize_config['scale_range']),
                probability=resize_config['probability']
            )
            
        logger.info(f"初始化攻击层: 启用攻击={list(self.attacks.keys())}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用攻击
        
        Args:
            x (torch.Tensor): 输入图像张量，形状 [B, C, H, W]
            
        Returns:
            torch.Tensor: 攻击后的图像张量
        """
        attacked_image = x
        applied_attacks = []
        
        # 随机应用攻击
        for attack_name, attack_module in self.attacks.items():
            original_image = attacked_image.clone()
            attacked_image = attack_module(attacked_image)
            
            # 检查是否实际应用了攻击
            if not torch.equal(original_image, attacked_image):
                applied_attacks.append(attack_name)
                
        logger.debug(f"应用的攻击: {applied_attacks}")
        
        return attacked_image
        
    def get_attack_info(self) -> Dict:
        """
        获取攻击配置信息
        
        Returns:
            Dict: 攻击配置信息
        """
        return {
            'enabled_attacks': list(self.attacks.keys()),
            'attack_config': self.attack_config
        }


if __name__ == "__main__":
    # 测试攻击层
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建测试数据
    batch_size = 2
    height, width = 64, 64
    channels = 4
    
    test_image = torch.randn(batch_size, channels, height, width)
    
    # 攻击配置
    attack_config = {
        'jpeg_compression': {
            'enabled': True,
            'quality_range': [50, 95],
            'probability': 0.7
        },
        'gaussian_noise': {
            'enabled': True,
            'sigma_range': [0.0, 0.1],
            'probability': 0.5
        },
        'resize_attack': {
            'enabled': True,
            'scale_range': [0.8, 1.2],
            'probability': 0.3
        }
    }
    
    # 创建攻击层
    attack_layer = AttackLayer(attack_config)
    
    # 打印攻击信息
    attack_info = attack_layer.get_attack_info()
    print("攻击层信息:")
    for key, value in attack_info.items():
        print(f"  {key}: {value}")
    
    # 测试攻击
    with torch.no_grad():
        attacked_image = attack_layer(test_image)
        
        # 计算差异
        diff = torch.abs(attacked_image - test_image)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        
        print(f"\n攻击测试结果:")
        print(f"  输入形状: {test_image.shape}")
        print(f"  输出形状: {attacked_image.shape}")
        print(f"  平均差异: {mean_diff:.6f}")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  输入范围: [{test_image.min().item():.4f}, {test_image.max().item():.4f}]")
        print(f"  输出范围: [{attacked_image.min().item():.4f}, {attacked_image.max().item():.4f}]")
    
    # 测试单个攻击
    print("\n单个攻击测试:")
    
    # JPEG压缩
    jpeg_attack = JPEGCompression(quality_range=(70, 90), probability=1.0)
    jpeg_result = jpeg_attack(test_image)
    jpeg_diff = torch.abs(jpeg_result - test_image).mean().item()
    print(f"  JPEG压缩差异: {jpeg_diff:.6f}")
    
    # 高斯噪声
    noise_attack = GaussianNoise(sigma_range=(0.05, 0.05), probability=1.0)
    noise_result = noise_attack(test_image)
    noise_diff = torch.abs(noise_result - test_image).mean().item()
    print(f"  高斯噪声差异: {noise_diff:.6f}")
    
    # 缩放攻击
    resize_attack = ResizeAttack(scale_range=(0.9, 0.9), probability=1.0)
    resize_result = resize_attack(test_image)
    resize_diff = torch.abs(resize_result - test_image).mean().item()
    print(f"  缩放攻击差异: {resize_diff:.6f}")
