"""
SAAH-WM Baseline 第二步 - 损失函数模块

本模块实现训练所需的各种损失函数，基于EditGuard的损失函数设计思想。

损失函数组成：
- L_total = λ_image * L_image + λ_robust * L_robust + λ_fragile * L_fragile

其中：
- L_image: 图像保真度损失（MSE + L1）
- L_robust: 鲁棒信息解码损失（BCE）
- L_fragile: 脆弱水印恢复损失（L1）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Tuple

# 配置日志
logger = logging.getLogger(__name__)


class ImageFidelityLoss(nn.Module):
    """
    图像保真度损失：确保嵌入水印后的图像与原图尽可能相似
    
    组合MSE损失和L1损失，既保证整体相似性又保证细节保真度。
    
    Args:
        mse_weight (float): MSE损失权重，默认1.0
        l1_weight (float): L1损失权重，默认1.0
    """
    
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 1.0):
        super(ImageFidelityLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        logger.info(f"初始化图像保真度损失: MSE权重={mse_weight}, L1权重={l1_weight}")
        
    def forward(self, watermarked_image: torch.Tensor, original_image: torch.Tensor) -> torch.Tensor:
        """
        计算图像保真度损失
        
        Args:
            watermarked_image (torch.Tensor): 嵌入水印的图像
            original_image (torch.Tensor): 原始图像
            
        Returns:
            torch.Tensor: 图像保真度损失
        """
        mse_loss = self.mse_loss(watermarked_image, original_image)
        l1_loss = self.l1_loss(watermarked_image, original_image)
        
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        logger.debug(f"图像保真度损失 - MSE: {mse_loss.item():.6f}, L1: {l1_loss.item():.6f}, "
                    f"总计: {total_loss.item():.6f}")
        
        return total_loss


class RobustMessageLoss(nn.Module):
    """
    鲁棒信息解码损失：确保嵌入的鲁棒信息在经过攻击后仍能被正确解码
    
    使用二元交叉熵损失，适合二进制信息包的解码任务。
    
    Args:
        loss_type (str): 损失类型，'bce'或'bce_logits'，默认'bce_logits'
        pos_weight (float): 正样本权重，用于处理类别不平衡，默认None
    """
    
    def __init__(self, loss_type: str = 'bce_logits', pos_weight: float = None):
        super(RobustMessageLoss, self).__init__()
        
        self.loss_type = loss_type
        
        if loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        elif loss_type == 'bce_logits':
            weight = torch.tensor(pos_weight) if pos_weight is not None else None
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
            
        logger.info(f"初始化鲁棒信息损失: 类型={loss_type}, 正样本权重={pos_weight}")
        
    def forward(self, decoded_message: torch.Tensor, target_message: torch.Tensor) -> torch.Tensor:
        """
        计算鲁棒信息解码损失
        
        Args:
            decoded_message (torch.Tensor): 解码的信息包
            target_message (torch.Tensor): 目标信息包
            
        Returns:
            torch.Tensor: 鲁棒信息解码损失
        """
        loss = self.loss_fn(decoded_message, target_message)
        
        logger.debug(f"鲁棒信息损失: {loss.item():.6f}")
        
        return loss
        
    def compute_bit_accuracy(self, decoded_message: torch.Tensor, target_message: torch.Tensor) -> float:
        """
        计算比特准确率
        
        Args:
            decoded_message (torch.Tensor): 解码的信息包
            target_message (torch.Tensor): 目标信息包
            
        Returns:
            float: 比特准确率
        """
        with torch.no_grad():
            if self.loss_type == 'bce_logits':
                # 对logits应用sigmoid
                decoded_prob = torch.sigmoid(decoded_message)
            else:
                decoded_prob = decoded_message
                
            # 二值化
            decoded_binary = (decoded_prob > 0.5).float()
            target_binary = (target_message > 0.5).float()
            
            # 计算准确率
            correct_bits = (decoded_binary == target_binary).float().sum()
            total_bits = target_binary.numel()
            accuracy = correct_bits / total_bits
            
            return accuracy.item()


class FragileRecoveryLoss(nn.Module):
    """
    脆弱水印恢复损失：确保脆弱编码器和解码器构成高保真的可逆变换
    
    使用L1损失确保恢复的图像和水印与原始版本尽可能接近。
    
    Args:
        image_weight (float): 图像恢复损失权重，默认1.0
        watermark_weight (float): 水印恢复损失权重，默认1.0
    """
    
    def __init__(self, image_weight: float = 1.0, watermark_weight: float = 1.0):
        super(FragileRecoveryLoss, self).__init__()
        
        self.image_weight = image_weight
        self.watermark_weight = watermark_weight
        
        self.l1_loss = nn.L1Loss()
        
        logger.info(f"初始化脆弱恢复损失: 图像权重={image_weight}, 水印权重={watermark_weight}")
        
    def forward(self, recovered_image: torch.Tensor, target_image: torch.Tensor,
                recovered_watermark: torch.Tensor, target_watermark: torch.Tensor) -> torch.Tensor:
        """
        计算脆弱水印恢复损失
        
        Args:
            recovered_image (torch.Tensor): 恢复的图像
            target_image (torch.Tensor): 目标图像
            recovered_watermark (torch.Tensor): 恢复的水印
            target_watermark (torch.Tensor): 目标水印
            
        Returns:
            torch.Tensor: 脆弱水印恢复损失
        """
        image_loss = self.l1_loss(recovered_image, target_image)
        watermark_loss = self.l1_loss(recovered_watermark, target_watermark)
        
        total_loss = self.image_weight * image_loss + self.watermark_weight * watermark_loss
        
        logger.debug(f"脆弱恢复损失 - 图像: {image_loss.item():.6f}, 水印: {watermark_loss.item():.6f}, "
                    f"总计: {total_loss.item():.6f}")
        
        return total_loss


class WatermarkLoss(nn.Module):
    """
    水印训练总损失函数
    
    组合所有子损失函数，实现多任务学习。
    
    Args:
        image_weight (float): 图像保真度损失权重，默认1.0
        robust_weight (float): 鲁棒信息损失权重，默认100.0
        fragile_weight (float): 脆弱恢复损失权重，默认10.0
    """
    
    def __init__(self, image_weight: float = 1.0, robust_weight: float = 100.0, 
                 fragile_weight: float = 10.0):
        super(WatermarkLoss, self).__init__()
        
        self.image_weight = image_weight
        self.robust_weight = robust_weight
        self.fragile_weight = fragile_weight
        
        # 初始化各个子损失函数
        self.image_loss = ImageFidelityLoss()
        self.robust_loss = RobustMessageLoss()
        self.fragile_loss = FragileRecoveryLoss()
        
        logger.info(f"初始化水印总损失: 图像权重={image_weight}, "
                   f"鲁棒权重={robust_weight}, 脆弱权重={fragile_weight}")
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            outputs (Dict[str, torch.Tensor]): 模型输出字典，包含：
                - 'watermarked_image': 嵌入水印的图像
                - 'decoded_message': 解码的信息包
                - 'recovered_image': 恢复的图像
                - 'recovered_watermark': 恢复的水印
            targets (Dict[str, torch.Tensor]): 目标字典，包含：
                - 'original_image': 原始图像
                - 'target_message': 目标信息包
                - 'target_watermark': 目标水印
                
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 总损失和各项损失详情
        """
        # 计算各项子损失
        l_image = self.image_loss(outputs['watermarked_image'], targets['original_image'])
        l_robust = self.robust_loss(outputs['decoded_message'], targets['target_message'])
        l_fragile = self.fragile_loss(outputs['recovered_image'], targets['original_image'],
                                     outputs['recovered_watermark'], targets['target_watermark'])
        
        # 计算总损失
        total_loss = (self.image_weight * l_image + 
                     self.robust_weight * l_robust + 
                     self.fragile_weight * l_fragile)
        
        # 计算比特准确率
        bit_accuracy = self.robust_loss.compute_bit_accuracy(
            outputs['decoded_message'], targets['target_message']
        )
        
        # 损失详情
        loss_details = {
            'total_loss': total_loss.item(),
            'image_loss': l_image.item(),
            'robust_loss': l_robust.item(),
            'fragile_loss': l_fragile.item(),
            'bit_accuracy': bit_accuracy
        }
        
        logger.debug(f"总损失计算完成: {loss_details}")
        
        return total_loss, loss_details
        
    def update_weights(self, image_weight: float = None, robust_weight: float = None, 
                      fragile_weight: float = None):
        """
        动态更新损失权重
        
        Args:
            image_weight (float): 新的图像损失权重
            robust_weight (float): 新的鲁棒损失权重
            fragile_weight (float): 新的脆弱损失权重
        """
        if image_weight is not None:
            self.image_weight = image_weight
        if robust_weight is not None:
            self.robust_weight = robust_weight
        if fragile_weight is not None:
            self.fragile_weight = fragile_weight
            
        logger.info(f"损失权重已更新: 图像={self.image_weight}, "
                   f"鲁棒={self.robust_weight}, 脆弱={self.fragile_weight}")


class PerceptualLoss(nn.Module):
    """
    感知损失：使用预训练VGG网络计算感知相似性
    
    Args:
        feature_layers (list): 使用的VGG特征层，默认['relu1_1', 'relu2_1', 'relu3_1']
        weights (list): 各层权重，默认[1.0, 1.0, 1.0]
    """
    
    def __init__(self, feature_layers: list = None, weights: list = None):
        super(PerceptualLoss, self).__init__()
        
        if feature_layers is None:
            feature_layers = ['relu1_1', 'relu2_1', 'relu3_1']
        if weights is None:
            weights = [1.0, 1.0, 1.0]
            
        self.feature_layers = feature_layers
        self.weights = weights
        
        # 这里可以集成VGG网络进行感知损失计算
        # 为简化实现，暂时使用MSE损失代替
        self.mse_loss = nn.MSELoss()
        
        logger.info(f"初始化感知损失: 特征层={feature_layers}, 权重={weights}")
        
    def forward(self, input_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            input_image (torch.Tensor): 输入图像
            target_image (torch.Tensor): 目标图像
            
        Returns:
            torch.Tensor: 感知损失
        """
        # 简化实现：使用MSE损失代替真正的感知损失
        # 在实际应用中，这里应该使用VGG特征进行计算
        loss = self.mse_loss(input_image, target_image)
        
        logger.debug(f"感知损失: {loss.item():.6f}")
        
        return loss


if __name__ == "__main__":
    # 测试损失函数
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建测试数据
    batch_size = 2
    height, width = 64, 64
    message_dim = 64
    
    # 模拟模型输出
    outputs = {
        'watermarked_image': torch.randn(batch_size, 4, height, width),
        'decoded_message': torch.randn(batch_size, message_dim),
        'recovered_image': torch.randn(batch_size, 4, height, width),
        'recovered_watermark': torch.randn(batch_size, 4, height, width)
    }
    
    # 模拟目标数据
    targets = {
        'original_image': torch.randn(batch_size, 4, height, width),
        'target_message': torch.randint(0, 2, (batch_size, message_dim)).float(),
        'target_watermark': torch.randn(batch_size, 4, height, width)
    }
    
    # 测试总损失函数
    criterion = WatermarkLoss(image_weight=1.0, robust_weight=100.0, fragile_weight=10.0)
    
    total_loss, loss_details = criterion(outputs, targets)
    
    print("损失函数测试结果:")
    for key, value in loss_details.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n总损失: {total_loss.item():.6f}")
    print(f"总损失形状: {total_loss.shape}")
    print(f"总损失需要梯度: {total_loss.requires_grad}")
