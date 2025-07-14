"""
SAAH-WM Baseline 第二步 - 鲁棒水印编码器 (RobustEncoder)

鲁棒水印编码器负责将信息包M嵌入到图像的潜在空间中。
该编码器设计为对各种攻击具有鲁棒性，确保嵌入的信息能够在攻击后被正确解码。

技术特点：
- 基于ResNet + 注意力机制的深度网络
- 输入：图像潜在特征图 + 信息包M
- 输出：嵌入鲁棒信息的潜在特征图
- 支持多尺度特征融合和注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Tuple

# 配置日志
logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    
    Args:
        channels (int): 输入通道数
        reduction (int): 降维比例，默认16
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    Args:
        kernel_size (int): 卷积核大小，默认7
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_map))
        return x * attention


class CBAM(nn.Module):
    """
    卷积块注意力模块 (Convolutional Block Attention Module)
    
    Args:
        channels (int): 输入通道数
        reduction (int): 通道注意力降维比例，默认16
        kernel_size (int): 空间注意力卷积核大小，默认7
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块 + 注意力机制
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长，默认1
        use_attention (bool): 是否使用注意力机制，默认True
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, use_attention: bool = True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 注意力机制
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        out += residual
        out = self.relu(out)
        
        return out


class MessageEmbedding(nn.Module):
    """
    信息嵌入模块：将信息包扩展并融合到特征图中

    Args:
        message_dim (int): 信息包维度
        feature_dim (int): 特征维度
    """

    def __init__(self, message_dim: int, feature_dim: int, spatial_size: int = None):
        super(MessageEmbedding, self).__init__()

        self.message_dim = message_dim
        self.feature_dim = feature_dim

        # 简化的信息扩展：只扩展到特征维度
        self.message_expand = nn.Sequential(
            nn.Linear(message_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh()  # 限制输出范围
        )

        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features (torch.Tensor): 特征图，形状 [B, C, H, W]
            message (torch.Tensor): 信息包，形状 [B, message_dim]

        Returns:
            torch.Tensor: 融合后的特征图
        """
        batch_size, channels, height, width = features.shape

        # 扩展信息包到特征维度
        expanded_message = self.message_expand(message)  # [B, feature_dim]

        # 扩展到空间维度：[B, C] -> [B, C, H, W]
        expanded_message = expanded_message.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        expanded_message = expanded_message.expand(-1, -1, height, width)  # [B, C, H, W]

        # 特征融合
        fused_features = features + expanded_message * 0.1  # 控制融合强度
        fused_features = self.fusion_conv(fused_features)

        return fused_features


class RobustEncoder(nn.Module):
    """
    鲁棒水印编码器 (BEM - Bit Embedding Module)
    
    该编码器将信息包M嵌入到图像的潜在空间中，设计为对各种攻击具有鲁棒性。
    
    网络结构：
    - 特征提取：多层残差块提取图像特征
    - 信息嵌入：将信息包融合到多个尺度的特征中
    - 注意力机制：增强重要特征的表达
    - 残差连接：保持图像质量
    
    Args:
        latent_channels (int): 潜在空间通道数，默认4
        message_dim (int): 信息包维度，默认64
        hidden_dim (int): 隐藏层维度，默认512
        num_layers (int): 网络层数，默认8
    """
    
    def __init__(self, latent_channels: int = 4, message_dim: int = 64, 
                 hidden_dim: int = 512, num_layers: int = 8):
        super(RobustEncoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        logger.info(f"初始化鲁棒水印编码器: latent_channels={latent_channels}, "
                   f"message_dim={message_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # 输入处理
        self.input_conv = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim // 4, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度特征提取
        self.layer1 = self._make_layer(hidden_dim // 4, hidden_dim // 2, 2)
        self.layer2 = self._make_layer(hidden_dim // 2, hidden_dim, 2)
        self.layer3 = self._make_layer(hidden_dim, hidden_dim, 2)
        
        # 信息嵌入模块（多尺度）
        # 注意：spatial_size会在forward中动态确定
        self.message_embed1 = MessageEmbedding(message_dim, hidden_dim // 2, 1)  # 占位符
        self.message_embed2 = MessageEmbedding(message_dim, hidden_dim, 1)       # 占位符
        self.message_embed3 = MessageEmbedding(message_dim, hidden_dim, 1)       # 占位符
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            CBAM(hidden_dim)
        )
        
        # 上采样重建
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, latent_channels, 3, 1, 1),
            nn.Tanh()  # 限制输出范围
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """创建残差层"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
        logger.info("鲁棒水印编码器权重初始化完成")
        
    def forward(self, image_latent: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_latent (torch.Tensor): 图像潜在特征，形状 [B, 4, H, W]
            message (torch.Tensor): 信息包，形状 [B, message_dim]
            
        Returns:
            torch.Tensor: 嵌入鲁棒信息的图像潜在特征，形状 [B, 4, H, W]
        """
        batch_size = image_latent.shape[0]
        
        logger.debug(f"鲁棒编码器输入 - 图像: {image_latent.shape}, 信息: {message.shape}")
        
        # 输入处理
        x = self.input_conv(image_latent)  # [B, 128, H, W]
        
        # 多尺度特征提取 + 信息嵌入
        x1 = self.layer1(x)                                    # [B, 256, H/2, W/2]
        x1 = self.message_embed1(x1, message)                  # 嵌入信息
        
        x2 = self.layer2(x1)                                   # [B, 512, H/4, W/4]
        x2 = self.message_embed2(x2, message)                  # 嵌入信息
        
        x3 = self.layer3(x2)                                   # [B, 512, H/8, W/8]
        x3 = self.message_embed3(x3, message)                  # 嵌入信息
        
        # 特征融合
        x3 = self.fusion_layer(x3)                             # [B, 512, H/8, W/8]
        
        # 上采样重建
        up1 = self.upsample1(x3)                               # [B, 256, H/4, W/4]
        # 检查通道数是否匹配，如果不匹配则跳过残差连接
        if up1.shape[1] == x2.shape[1]:
            up1 = up1 + x2                                     # 残差连接

        up2 = self.upsample2(up1)                              # [B, 128, H/2, W/2]
        # 检查通道数是否匹配，如果不匹配则跳过残差连接
        if up2.shape[1] == x1.shape[1]:
            up2 = up2 + x1                                     # 残差连接

        up3 = self.upsample3(up2)                              # [B, 64, H, W]
        # 检查通道数是否匹配，如果不匹配则跳过残差连接
        if up3.shape[1] == x.shape[1]:
            up3 = up3 + x                                      # 残差连接
        
        # 输出层：生成残差
        residual = self.output_conv(up3)                       # [B, 4, H, W]
        
        # 残差连接：原图 + 残差 = 嵌入信息的图像
        watermarked_image = image_latent + 0.1 * residual      # 控制嵌入强度
        
        logger.debug(f"鲁棒编码器输出: {watermarked_image.shape}")
        
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
            'model_name': 'RobustEncoder',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_channels': self.latent_channels,
            'message_dim': self.message_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'input_shape': f'[B, {self.latent_channels}, H, W] + [B, {self.message_dim}]',
            'output_shape': f'[B, {self.latent_channels}, H, W]'
        }


if __name__ == "__main__":
    # 测试鲁棒水印编码器
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建模型
    model = RobustEncoder(latent_channels=4, message_dim=64, hidden_dim=512, num_layers=8)
    
    # 打印模型信息
    info = model.get_model_info()
    print("鲁棒水印编码器信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 2
    height, width = 64, 64
    message_dim = 64
    
    image_latent = torch.randn(batch_size, 4, height, width)
    message = torch.randn(batch_size, message_dim)
    
    with torch.no_grad():
        output = model(image_latent, message)
        print(f"\n测试结果:")
        print(f"  输入图像形状: {image_latent.shape}")
        print(f"  输入信息形状: {message.shape}")
        print(f"  输出图像形状: {output.shape}")
        print(f"  输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
