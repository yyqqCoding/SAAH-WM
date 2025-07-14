"""
SAAH-WM Baseline 第二步 - 脆弱水印编码器 (FragileEncoder)

脆弱水印编码器负责将基准水印W_base嵌入到原始图像的潜在空间中。
该编码器实现可逆变换，一旦图像被篡改，变换就会被破坏，从而实现篡改检测。

技术特点：
- 基于U-Net架构的可逆网络设计
- 输入：原始潜在特征图 + 基准水印W_base
- 输出：嵌入脆弱水印的潜在特征图
- 支持高保真度的可逆变换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

# 配置日志
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    卷积块：卷积 + 实例归一化 + ReLU激活
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小，默认3
        stride (int): 步长，默认1
        padding (int): 填充，默认1
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    """
    下采样块：步长为2的卷积 + 卷积块
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        num_blocks (int): 卷积块数量，默认2
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super(DownBlock, self).__init__()
        
        # 下采样卷积
        self.downsample = ConvBlock(in_channels, in_channels, stride=2)
        
        # 后续卷积块
        blocks = []
        blocks.append(ConvBlock(in_channels, out_channels))
        for _ in range(num_blocks - 1):
            blocks.append(ConvBlock(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class UpBlock(nn.Module):
    """
    上采样块：双线性插值上采样 + 卷积块
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 双线性插值上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块：用于特征融合和处理
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        num_blocks (int): 残差块数量，默认2
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super(ResidualBlock, self).__init__()
        
        blocks = []
        blocks.append(ConvBlock(in_channels, out_channels))
        for _ in range(num_blocks - 1):
            blocks.append(ConvBlock(out_channels, out_channels))
            
        self.blocks = nn.Sequential(*blocks)
        
        # 如果输入输出通道数不同，需要调整残差连接
        self.shortcut = nn.Identity() if in_channels == out_channels else \
                       nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = self.shortcut(x)
        x = self.blocks(x)
        return x + residual


class FragileEncoder(nn.Module):
    """
    脆弱水印编码器 (IHM - Image Hiding Module)
    
    该编码器将基准水印W_base嵌入到原始图像的潜在空间中，实现可逆变换。
    一旦图像被篡改，这个可逆变换就会被破坏，从而实现篡改检测。
    
    网络结构：
    - 编码器：多层下采样提取特征
    - 水印融合：将基准水印信息融合到特征中
    - 解码器：多层上采样重建图像
    - 残差连接：保持高保真度
    
    Args:
        latent_channels (int): 潜在空间通道数，默认4（Stable Diffusion）
        hidden_dim (int): 隐藏层维度，默认256
        num_layers (int): 网络层数，默认6
    """
    
    def __init__(self, latent_channels: int = 4, hidden_dim: int = 256, num_layers: int = 6):
        super(FragileEncoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        logger.info(f"初始化脆弱水印编码器: latent_channels={latent_channels}, "
                   f"hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # 输入处理：原始图像(4通道) + 基准水印(4通道) = 8通道
        self.input_conv = ConvBlock(latent_channels * 2, hidden_dim // 4)
        
        # 编码器：下采样路径
        self.down1 = DownBlock(hidden_dim // 4, hidden_dim // 2)  # 64x64 -> 32x32
        self.down2 = DownBlock(hidden_dim // 2, hidden_dim)       # 32x32 -> 16x16
        self.down3 = DownBlock(hidden_dim, hidden_dim * 2)        # 16x16 -> 8x8
        
        # 瓶颈层：特征处理
        self.bottleneck = ResidualBlock(hidden_dim * 2, hidden_dim * 2, num_blocks=3)
        
        # 解码器：上采样路径
        self.up3 = UpBlock(hidden_dim * 2, hidden_dim)            # 8x8 -> 16x16
        self.att3 = ResidualBlock(hidden_dim * 2, hidden_dim)     # 跳跃连接融合
        
        self.up2 = UpBlock(hidden_dim, hidden_dim // 2)           # 16x16 -> 32x32
        self.att2 = ResidualBlock(hidden_dim, hidden_dim // 2)    # 跳跃连接融合
        
        self.up1 = UpBlock(hidden_dim // 2, hidden_dim // 4)      # 32x32 -> 64x64
        self.att1 = ResidualBlock(hidden_dim // 2, hidden_dim // 4)  # 跳跃连接融合
        
        # 输出层：生成嵌入水印的图像
        self.output_conv = nn.Conv2d(hidden_dim // 4, latent_channels, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
        logger.info("脆弱水印编码器权重初始化完成")
        
    def forward(self, image_latent: torch.Tensor, base_watermark: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_latent (torch.Tensor): 原始图像潜在特征，形状 [B, 4, H, W]
            base_watermark (torch.Tensor): 基准水印，形状 [B, 4, H, W]
            
        Returns:
            torch.Tensor: 嵌入脆弱水印的图像潜在特征，形状 [B, 4, H, W]
        """
        batch_size = image_latent.shape[0]
        
        logger.debug(f"脆弱编码器输入 - 图像: {image_latent.shape}, 水印: {base_watermark.shape}")
        
        # 输入融合：将图像和基准水印拼接
        x = torch.cat([image_latent, base_watermark], dim=1)  # [B, 8, H, W]
        
        # 输入处理
        x = self.input_conv(x)  # [B, 64, H, W]
        
        # 编码器：下采样提取特征
        d1 = self.down1(x)      # [B, 128, H/2, W/2]
        d2 = self.down2(d1)     # [B, 256, H/4, W/4]
        d3 = self.down3(d2)     # [B, 512, H/8, W/8]
        
        # 瓶颈层：特征处理
        bottleneck = self.bottleneck(d3)  # [B, 512, H/8, W/8]
        
        # 解码器：上采样重建
        u3 = self.up3(bottleneck)                    # [B, 256, H/4, W/4]
        u3 = torch.cat([d2, u3], dim=1)             # 跳跃连接
        u3 = self.att3(u3)                          # [B, 256, H/4, W/4]
        
        u2 = self.up2(u3)                           # [B, 128, H/2, W/2]
        u2 = torch.cat([d1, u2], dim=1)             # 跳跃连接
        u2 = self.att2(u2)                          # [B, 128, H/2, W/2]
        
        u1 = self.up1(u2)                           # [B, 64, H, W]
        u1 = torch.cat([x, u1], dim=1)              # 跳跃连接
        u1 = self.att1(u1)                          # [B, 64, H, W]
        
        # 输出层：生成残差
        residual = self.output_conv(u1)              # [B, 4, H, W]
        
        # 残差连接：原图 + 残差 = 嵌入水印的图像
        watermarked_image = image_latent + residual
        
        logger.debug(f"脆弱编码器输出: {watermarked_image.shape}")
        
        return watermarked_image
        
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 包含模型参数和结构信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'FragileEncoder',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_channels': self.latent_channels,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'input_shape': f'[B, {self.latent_channels * 2}, H, W]',
            'output_shape': f'[B, {self.latent_channels}, H, W]'
        }


if __name__ == "__main__":
    # 测试脆弱水印编码器
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建模型
    model = FragileEncoder(latent_channels=4, hidden_dim=256, num_layers=6)
    
    # 打印模型信息
    info = model.get_model_info()
    print("脆弱水印编码器信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 2
    height, width = 64, 64
    
    image_latent = torch.randn(batch_size, 4, height, width)
    base_watermark = torch.randn(batch_size, 4, height, width)
    
    with torch.no_grad():
        output = model(image_latent, base_watermark)
        print(f"\n测试结果:")
        print(f"  输入图像形状: {image_latent.shape}")
        print(f"  基准水印形状: {base_watermark.shape}")
        print(f"  输出图像形状: {output.shape}")
        print(f"  输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
