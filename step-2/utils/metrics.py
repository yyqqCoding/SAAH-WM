"""
SAAH-WM Baseline 第二步 - 评估指标模块

本模块实现训练和验证过程中使用的各种评估指标。

主要指标：
- PSNR (峰值信噪比): 图像质量评估
- SSIM (结构相似性): 图像结构相似性评估  
- LPIPS (感知损失): 感知相似性评估
- Bit Accuracy (比特准确率): 信息解码准确率
- Message Recovery Rate (信息恢复率): 完整信息包恢复率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, List
import math

# 配置日志
logger = logging.getLogger(__name__)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        img1 (torch.Tensor): 第一张图像
        img2 (torch.Tensor): 第二张图像  
        max_val (float): 图像的最大值，默认1.0
        
    Returns:
        float: PSNR值（dB）
    """
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
        
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
                  size_average: bool = True) -> float:
    """
    计算结构相似性指数 (SSIM)

    Args:
        img1 (torch.Tensor): 第一张图像，形状 [B, C, H, W]
        img2 (torch.Tensor): 第二张图像，形状 [B, C, H, W]
        window_size (int): 窗口大小，默认11
        size_average (bool): 是否对所有像素求平均，默认True

    Returns:
        float: SSIM值 [0, 1]
    """
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        """创建高斯窗口"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(1) * g.unsqueeze(0)

    # 获取通道数
    channels = img1.shape[1]

    # 创建高斯窗口
    window = gaussian_window(window_size).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    # 常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_bit_accuracy(predicted: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算比特准确率
    
    Args:
        predicted (torch.Tensor): 预测的信息包，形状 [B, message_dim]
        target (torch.Tensor): 目标信息包，形状 [B, message_dim]
        threshold (float): 二值化阈值，默认0.5
        
    Returns:
        float: 比特准确率 [0, 1]
    """
    with torch.no_grad():
        # 二值化
        pred_binary = (predicted > threshold).float()
        target_binary = (target > threshold).float()
        
        # 计算准确率
        correct_bits = (pred_binary == target_binary).float().sum()
        total_bits = target_binary.numel()
        accuracy = correct_bits / total_bits
        
        return accuracy.item()


def calculate_message_recovery_rate(predicted: torch.Tensor, target: torch.Tensor, 
                                  threshold: float = 0.5) -> float:
    """
    计算完整信息包恢复率
    
    Args:
        predicted (torch.Tensor): 预测的信息包，形状 [B, message_dim]
        target (torch.Tensor): 目标信息包，形状 [B, message_dim]
        threshold (float): 二值化阈值，默认0.5
        
    Returns:
        float: 信息包恢复率 [0, 1]
    """
    with torch.no_grad():
        # 二值化
        pred_binary = (predicted > threshold).float()
        target_binary = (target > threshold).float()
        
        # 计算每个样本的准确率
        batch_size = predicted.shape[0]
        correct_messages = 0
        
        for i in range(batch_size):
            if torch.equal(pred_binary[i], target_binary[i]):
                correct_messages += 1
                
        recovery_rate = correct_messages / batch_size
        return recovery_rate


class WatermarkMetrics:
    """
    水印系统评估指标计算器
    
    集成所有评估指标的计算，提供统一的接口。
    """
    
    def __init__(self):
        self.reset()
        logger.info("初始化水印评估指标计算器")
        
    def reset(self):
        """重置所有累积指标"""
        self.total_samples = 0
        self.cumulative_metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'bit_accuracy': 0.0,
            'message_recovery_rate': 0.0,
            'fragile_psnr': 0.0,
            'fragile_ssim': 0.0
        }
        
    def update(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        更新指标
        
        Args:
            outputs (Dict[str, torch.Tensor]): 模型输出
            targets (Dict[str, torch.Tensor]): 目标数据
            
        Returns:
            Dict[str, float]: 当前批次的指标
        """
        batch_size = outputs['watermarked_image'].shape[0]
        
        # 计算图像保真度指标
        psnr = calculate_psnr(outputs['watermarked_image'], targets['original_image'])
        ssim = calculate_ssim(outputs['watermarked_image'], targets['original_image'])
        
        # 计算信息解码指标
        bit_acc = calculate_bit_accuracy(outputs['decoded_message'], targets['target_message'])
        msg_recovery = calculate_message_recovery_rate(outputs['decoded_message'], targets['target_message'])
        
        # 计算脆弱水印恢复指标
        fragile_psnr = calculate_psnr(outputs['recovered_image'], targets['original_image'])
        fragile_ssim = calculate_ssim(outputs['recovered_image'], targets['original_image'])
        
        # 当前批次指标
        current_metrics = {
            'psnr': psnr,
            'ssim': ssim,
            'bit_accuracy': bit_acc,
            'message_recovery_rate': msg_recovery,
            'fragile_psnr': fragile_psnr,
            'fragile_ssim': fragile_ssim
        }
        
        # 累积指标
        self.total_samples += batch_size
        for key, value in current_metrics.items():
            self.cumulative_metrics[key] += value * batch_size
            
        logger.debug(f"指标更新完成: {current_metrics}")
        
        return current_metrics
        
    def compute_average(self) -> Dict[str, float]:
        """
        计算平均指标
        
        Returns:
            Dict[str, float]: 平均指标
        """
        if self.total_samples == 0:
            return {key: 0.0 for key in self.cumulative_metrics.keys()}
            
        average_metrics = {}
        for key, value in self.cumulative_metrics.items():
            average_metrics[key] = value / self.total_samples
            
        return average_metrics
        
    def check_performance_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        检查性能是否达到阈值要求
        
        Args:
            thresholds (Dict[str, float]): 性能阈值
            
        Returns:
            Dict[str, bool]: 各指标是否达到阈值
        """
        average_metrics = self.compute_average()
        results = {}
        
        for metric, threshold in thresholds.items():
            if metric in average_metrics:
                results[metric] = average_metrics[metric] >= threshold
            else:
                results[metric] = False
                
        return results
        
    def get_summary(self) -> str:
        """
        获取指标摘要字符串
        
        Returns:
            str: 格式化的指标摘要
        """
        average_metrics = self.compute_average()
        
        summary = f"评估指标摘要 (样本数: {self.total_samples}):\n"
        summary += f"  图像保真度 - PSNR: {average_metrics['psnr']:.2f}dB, SSIM: {average_metrics['ssim']:.4f}\n"
        summary += f"  信息解码 - 比特准确率: {average_metrics['bit_accuracy']:.4f}, 恢复率: {average_metrics['message_recovery_rate']:.4f}\n"
        summary += f"  脆弱恢复 - PSNR: {average_metrics['fragile_psnr']:.2f}dB, SSIM: {average_metrics['fragile_ssim']:.4f}"
        
        return summary


class PerformanceMonitor:
    """
    性能监控器：跟踪训练过程中的性能变化
    """
    
    def __init__(self, save_best: bool = True):
        self.save_best = save_best
        self.best_metrics = {}
        self.history = []
        
        logger.info(f"初始化性能监控器: 保存最佳={save_best}")
        
    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        更新性能记录
        
        Args:
            epoch (int): 当前epoch
            metrics (Dict[str, float]): 当前指标
            
        Returns:
            bool: 是否达到新的最佳性能
        """
        # 添加到历史记录
        record = {'epoch': epoch, **metrics}
        self.history.append(record)
        
        # 检查是否是最佳性能
        is_best = False
        if self.save_best:
            # 以比特准确率作为主要指标
            current_score = metrics.get('bit_accuracy', 0.0)
            best_score = self.best_metrics.get('bit_accuracy', 0.0)
            
            if current_score > best_score:
                self.best_metrics = metrics.copy()
                self.best_metrics['epoch'] = epoch
                is_best = True
                
                logger.info(f"发现新的最佳性能 (Epoch {epoch}): 比特准确率={current_score:.4f}")
                
        return is_best
        
    def get_best_metrics(self) -> Dict[str, float]:
        """获取最佳指标"""
        return self.best_metrics.copy()
        
    def get_history(self) -> List[Dict[str, float]]:
        """获取历史记录"""
        return self.history.copy()
        
    def save_history(self, filepath: str):
        """保存历史记录到文件"""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
            
        logger.info(f"性能历史记录已保存到: {filepath}")


if __name__ == "__main__":
    # 测试评估指标
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建测试数据
    batch_size = 4
    height, width = 64, 64
    message_dim = 64
    
    # 模拟模型输出
    outputs = {
        'watermarked_image': torch.randn(batch_size, 4, height, width),
        'decoded_message': torch.sigmoid(torch.randn(batch_size, message_dim)),  # 概率值
        'recovered_image': torch.randn(batch_size, 4, height, width),
        'recovered_watermark': torch.randn(batch_size, 4, height, width)
    }
    
    # 模拟目标数据
    targets = {
        'original_image': torch.randn(batch_size, 4, height, width),
        'target_message': torch.randint(0, 2, (batch_size, message_dim)).float(),
        'target_watermark': torch.randn(batch_size, 4, height, width)
    }
    
    # 测试单个指标计算
    print("单个指标测试:")
    
    psnr = calculate_psnr(outputs['watermarked_image'], targets['original_image'])
    print(f"  PSNR: {psnr:.2f} dB")
    
    ssim = calculate_ssim(outputs['watermarked_image'], targets['original_image'])
    print(f"  SSIM: {ssim:.4f}")
    
    bit_acc = calculate_bit_accuracy(outputs['decoded_message'], targets['target_message'])
    print(f"  比特准确率: {bit_acc:.4f}")
    
    msg_recovery = calculate_message_recovery_rate(outputs['decoded_message'], targets['target_message'])
    print(f"  信息恢复率: {msg_recovery:.4f}")
    
    # 测试指标计算器
    print("\n指标计算器测试:")
    
    metrics_calculator = WatermarkMetrics()
    
    # 更新指标
    current_metrics = metrics_calculator.update(outputs, targets)
    print("当前批次指标:")
    for key, value in current_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 计算平均指标
    average_metrics = metrics_calculator.compute_average()
    print("\n平均指标:")
    for key, value in average_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 检查性能阈值
    thresholds = {
        'psnr': 38.0,
        'ssim': 0.95,
        'bit_accuracy': 0.995
    }
    
    threshold_results = metrics_calculator.check_performance_thresholds(thresholds)
    print("\n阈值检查结果:")
    for metric, passed in threshold_results.items():
        status = "✅ 通过" if passed else "❌ 未达到"
        print(f"  {metric}: {status}")
    
    # 打印摘要
    print(f"\n{metrics_calculator.get_summary()}")
    
    # 测试性能监控器
    print("\n性能监控器测试:")
    
    monitor = PerformanceMonitor(save_best=True)
    
    # 模拟几个epoch的性能
    for epoch in range(1, 4):
        # 模拟性能提升
        fake_metrics = {
            'psnr': 35.0 + epoch * 2,
            'ssim': 0.90 + epoch * 0.02,
            'bit_accuracy': 0.85 + epoch * 0.05,
            'message_recovery_rate': 0.80 + epoch * 0.05
        }
        
        is_best = monitor.update(epoch, fake_metrics)
        print(f"  Epoch {epoch}: 是否最佳={is_best}")
    
    best_metrics = monitor.get_best_metrics()
    print(f"\n最佳性能 (Epoch {best_metrics.get('epoch', 'N/A')}):")
    for key, value in best_metrics.items():
        if key != 'epoch':
            print(f"  {key}: {value:.4f}")
