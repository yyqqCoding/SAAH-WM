"""
SAAH-WM Baseline 第二步 - 鲁棒信息解码器 (RobustDecoder)

鲁棒信息解码器负责从被攻击的图像中解码出嵌入的信息包M。
该解码器设计为对各种攻击具有鲁棒性，能够在图像受到攻击后仍然正确解码信息。

技术特点：
- 基于CNN + 全连接层的深度网络
- 输入：被攻击的图像潜在特征图
- 输出：解码的信息包M
- 支持多尺度特征提取和鲁棒性设计
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
    卷积块：卷积 + 批归一化 + ReLU激活
    
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
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        x = self.bn(x)
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


class ResidualBlock(nn.Module):
    """
    残差块：用于增强特征表达能力
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长，默认1
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
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
        
        out += residual
        out = self.relu(out)
        
        return out


class GlobalAveragePooling(nn.Module):
    """
    全局平均池化：将特征图转换为特征向量
    """
    
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class RobustDecoder(nn.Module):
    """
    鲁棒信息解码器 (BRM - Bit Recovery Module)
    
    该解码器从被攻击的图像中解码出嵌入的信息包M，设计为对各种攻击具有鲁棒性。
    
    网络结构：
    - 特征提取：多层卷积和下采样提取图像特征
    - 残差连接：增强特征表达能力
    - 全局池化：将空间特征转换为全局特征
    - 全连接层：解码出信息包
    
    Args:
        latent_channels (int): 潜在空间通道数，默认4
        message_dim (int): 信息包维度，默认64
        hidden_dim (int): 隐藏层维度，默认512
        num_layers (int): 网络层数，默认6
    """
    
    def __init__(self, latent_channels: int = 4, message_dim: int = 64, 
                 hidden_dim: int = 512, num_layers: int = 6):
        super(RobustDecoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        logger.info(f"初始化鲁棒信息解码器: latent_channels={latent_channels}, "
                   f"message_dim={message_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # 输入处理
        self.input_conv = ConvBlock(latent_channels, hidden_dim // 8)
        
        # 特征提取网络：多层下采样
        self.down1 = DownBlock(hidden_dim // 8, hidden_dim // 4)    # 64x64 -> 32x32
        self.down2 = DownBlock(hidden_dim // 4, hidden_dim // 2)    # 32x32 -> 16x16
        self.down3 = DownBlock(hidden_dim // 2, hidden_dim)         # 16x16 -> 8x8
        self.down4 = DownBlock(hidden_dim, hidden_dim)              # 8x8 -> 4x4
        
        # 残差块：增强特征表达
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        
        # 全局特征提取
        self.global_pool = GlobalAveragePooling()
        
        # 信息解码网络
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 8, message_dim),
            nn.Sigmoid()  # 输出概率值 [0, 1]
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
        logger.info("鲁棒信息解码器权重初始化完成")
        
    def forward(self, image_latent: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_latent (torch.Tensor): 被攻击的图像潜在特征，形状 [B, 4, H, W]
            
        Returns:
            torch.Tensor: 解码的信息包，形状 [B, message_dim]
        """
        batch_size = image_latent.shape[0]
        
        logger.debug(f"鲁棒解码器输入: {image_latent.shape}")
        
        # 输入处理
        x = self.input_conv(image_latent)  # [B, 64, H, W]
        
        # 多层特征提取
        x = self.down1(x)                  # [B, 128, H/2, W/2]
        x = self.down2(x)                  # [B, 256, H/4, W/4]
        x = self.down3(x)                  # [B, 512, H/8, W/8]
        x = self.down4(x)                  # [B, 512, H/16, W/16]
        
        # 残差块增强特征
        x = self.res_blocks(x)             # [B, 512, H/16, W/16]
        
        # 全局特征提取
        global_features = self.global_pool(x)  # [B, 512]
        
        # 信息解码
        decoded_message = self.decoder(global_features)  # [B, message_dim]
        
        logger.debug(f"鲁棒解码器输出: {decoded_message.shape}")
        
        return decoded_message
        
    def decode_binary_message(self, image_latent: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        解码二进制信息包
        
        Args:
            image_latent (torch.Tensor): 被攻击的图像潜在特征
            threshold (float): 二值化阈值，默认0.5
            
        Returns:
            torch.Tensor: 二进制信息包，形状 [B, message_dim]
        """
        with torch.no_grad():
            prob_message = self.forward(image_latent)
            binary_message = (prob_message > threshold).float()
            return binary_message
            
    def get_bit_accuracy(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算比特准确率
        
        Args:
            predicted (torch.Tensor): 预测的信息包
            target (torch.Tensor): 目标信息包
            
        Returns:
            float: 比特准确率
        """
        with torch.no_grad():
            # 二值化预测结果
            predicted_binary = (predicted > 0.5).float()
            target_binary = (target > 0.5).float()
            
            # 计算准确率
            correct_bits = (predicted_binary == target_binary).float().sum()
            total_bits = target_binary.numel()
            accuracy = correct_bits / total_bits
            
            return accuracy.item()
        
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 包含模型参数和结构信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'RobustDecoder',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_channels': self.latent_channels,
            'message_dim': self.message_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'input_shape': f'[B, {self.latent_channels}, H, W]',
            'output_shape': f'[B, {self.message_dim}]'
        }


if __name__ == "__main__":
    # 测试鲁棒信息解码器
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建模型
    model = RobustDecoder(latent_channels=4, message_dim=64, hidden_dim=512, num_layers=6)
    
    # 打印模型信息
    info = model.get_model_info()
    print("鲁棒信息解码器信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 2
    height, width = 64, 64
    message_dim = 64
    
    image_latent = torch.randn(batch_size, 4, height, width)
    target_message = torch.randint(0, 2, (batch_size, message_dim)).float()
    
    with torch.no_grad():
        # 解码信息
        decoded_message = model(image_latent)
        binary_message = model.decode_binary_message(image_latent)
        
        # 计算准确率
        accuracy = model.get_bit_accuracy(decoded_message, target_message)
        
        print(f"\n测试结果:")
        print(f"  输入图像形状: {image_latent.shape}")
        print(f"  解码信息形状: {decoded_message.shape}")
        print(f"  二进制信息形状: {binary_message.shape}")
        print(f"  解码数值范围: [{decoded_message.min().item():.4f}, {decoded_message.max().item():.4f}]")
        print(f"  随机准确率: {accuracy:.4f}")
        print(f"  二进制信息示例: {binary_message[0][:10].int().tolist()}")
