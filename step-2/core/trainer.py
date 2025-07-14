"""
SAAH-WM Baseline 第二步 - 主训练器

本模块实现完整的训练流程，协调所有组件：
- 四个网络模型的训练
- 损失函数计算
- 攻击模拟
- 指标评估
- 模型保存和加载

训练流程：
1. 数据加载
2. 前向传播（编码器 -> 攻击 -> 解码器）
3. 损失计算
4. 反向传播
5. 指标评估
6. 模型保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

# 导入模型和工具
from models import FragileEncoder, RobustEncoder, RobustDecoder, FragileDecoder
from utils.loss_functions import WatermarkLoss
from utils.attack_layers import AttackLayer
from utils.metrics import WatermarkMetrics, PerformanceMonitor
from utils.logger_config import TrainingLogger

# 配置日志
logger = logging.getLogger(__name__)


class WatermarkTrainer:
    """
    水印训练器
    
    负责整个训练流程的控制和协调。
    
    Args:
        config (Dict): 训练配置字典
        device (str): 训练设备，'cuda'或'cpu'
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.global_step = 0
        
        logger.info(f"初始化水印训练器: 设备={device}")
        
        # 初始化模型
        self._setup_models()
        
        # 初始化损失函数和攻击层
        self._setup_loss_and_attacks()
        
        # 初始化优化器和调度器
        self._setup_optimizers()
        
        # 初始化评估工具
        self._setup_evaluation_tools()
        
        # 创建保存目录
        self._setup_directories()
        
        logger.info("水印训练器初始化完成")
        
    def _setup_models(self):
        """初始化网络模型"""
        model_config = self.config['model']
        
        # 创建四个核心模型
        self.fragile_encoder = FragileEncoder(
            latent_channels=model_config['latent_channels'],
            hidden_dim=model_config['fragile_encoder']['hidden_dim'],
            num_layers=model_config['fragile_encoder']['num_layers']
        ).to(self.device)
        
        self.robust_encoder = RobustEncoder(
            latent_channels=model_config['latent_channels'],
            message_dim=model_config['robust_encoder']['message_dim'],
            hidden_dim=model_config['robust_encoder']['hidden_dim'],
            num_layers=model_config['robust_encoder']['num_layers']
        ).to(self.device)
        
        self.robust_decoder = RobustDecoder(
            latent_channels=model_config['latent_channels'],
            message_dim=model_config['robust_decoder']['message_dim'],
            hidden_dim=model_config['robust_decoder']['hidden_dim'],
            num_layers=model_config['robust_decoder']['num_layers']
        ).to(self.device)
        
        self.fragile_decoder = FragileDecoder(
            latent_channels=model_config['latent_channels'],
            hidden_dim=model_config['fragile_decoder']['hidden_dim'],
            num_layers=model_config['fragile_decoder']['num_layers']
        ).to(self.device)
        
        # 打印模型信息
        models = {
            'FragileEncoder': self.fragile_encoder,
            'RobustEncoder': self.robust_encoder,
            'RobustDecoder': self.robust_decoder,
            'FragileDecoder': self.fragile_decoder
        }
        
        total_params = 0
        for name, model in models.items():
            info = model.get_model_info()
            logger.info(f"{name}: {info['total_parameters']:,} 参数")
            total_params += info['total_parameters']
            
        logger.info(f"总参数量: {total_params:,}")
        
    def _setup_loss_and_attacks(self):
        """初始化损失函数和攻击层"""
        loss_weights = self.config['training']['loss_weights']
        
        # 损失函数
        self.criterion = WatermarkLoss(
            image_weight=loss_weights['image_loss'],
            robust_weight=loss_weights['robust_loss'],
            fragile_weight=loss_weights['fragile_loss']
        ).to(self.device)
        
        # 攻击层
        self.attack_layer = AttackLayer(self.config['attacks']).to(self.device)
        
        logger.info("损失函数和攻击层初始化完成")
        
    def _setup_optimizers(self):
        """初始化优化器和学习率调度器"""
        training_config = self.config['training']
        
        # 收集所有模型参数
        all_parameters = []
        all_parameters.extend(self.fragile_encoder.parameters())
        all_parameters.extend(self.robust_encoder.parameters())
        all_parameters.extend(self.robust_decoder.parameters())
        all_parameters.extend(self.fragile_decoder.parameters())
        
        # 优化器
        if training_config['optimizer'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                all_parameters,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=(training_config['beta1'], training_config['beta2']),
                eps=training_config['eps']
            )
        else:
            raise ValueError(f"不支持的优化器: {training_config['optimizer']}")
        
        # 学习率调度器
        if training_config['lr_scheduler'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config['min_lr']
            )
        else:
            self.scheduler = None
            
        logger.info(f"优化器和调度器初始化完成: {training_config['optimizer']}")
        
    def _setup_evaluation_tools(self):
        """初始化评估工具"""
        self.metrics_calculator = WatermarkMetrics()
        self.performance_monitor = PerformanceMonitor(save_best=True)
        
        logger.info("评估工具初始化完成")
        
    def _setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.config['checkpoints']['save_dir'],
            self.config['logging']['log_dir'],
            'outputs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("目录结构创建完成")
        
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch (Dict[str, torch.Tensor]): 输入批次数据
            
        Returns:
            Dict[str, torch.Tensor]: 模型输出
        """
        # 获取输入数据
        image_latent = batch['image_latent'].to(self.device)      # [B, 4, H, W]
        base_watermark = batch['base_watermark'].to(self.device)  # [B, 4, H, W]
        message_bits = batch['message_bits'].to(self.device)      # [B, message_dim]
        
        batch_size = image_latent.shape[0]
        
        # 步骤1: 脆弱水印嵌入
        watermarked_fragile = self.fragile_encoder(image_latent, base_watermark)
        
        # 步骤2: 鲁棒信息嵌入
        watermarked_robust = self.robust_encoder(watermarked_fragile, message_bits)
        
        # 步骤3: 攻击模拟（仅对鲁棒解码器的输入）
        attacked_image = self.attack_layer(watermarked_robust)
        
        # 步骤4: 鲁棒信息解码
        decoded_message = self.robust_decoder(attacked_image)
        
        # 步骤5: 脆弱水印解码（使用未攻击的图像）
        recovered_image, recovered_watermark = self.fragile_decoder(watermarked_robust)
        
        # 构建输出
        outputs = {
            'watermarked_image': watermarked_robust,      # 最终嵌入水印的图像
            'attacked_image': attacked_image,             # 被攻击的图像
            'decoded_message': decoded_message,           # 解码的信息包
            'recovered_image': recovered_image,           # 恢复的原始图像
            'recovered_watermark': recovered_watermark    # 恢复的基准水印
        }
        
        return outputs
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            outputs (Dict[str, torch.Tensor]): 模型输出
            targets (Dict[str, torch.Tensor]): 目标数据
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 总损失和损失详情
        """
        return self.criterion(outputs, targets)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch (Dict[str, torch.Tensor]): 输入批次
            
        Returns:
            Dict[str, float]: 训练指标
        """
        # 设置为训练模式
        self.fragile_encoder.train()
        self.robust_encoder.train()
        self.robust_decoder.train()
        self.fragile_decoder.train()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.forward_pass(batch)
        
        # 构建目标数据
        targets = {
            'original_image': batch['image_latent'].to(self.device),
            'target_message': batch['message_bits'].to(self.device),
            'target_watermark': batch['base_watermark'].to(self.device)
        }
        
        # 计算损失
        total_loss, loss_details = self.compute_loss(outputs, targets)
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        if self.config['training'].get('gradient_clip_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.get_all_parameters(),
                self.config['training']['gradient_clip_norm']
            )
        
        # 更新参数
        self.optimizer.step()
        
        # 更新全局步数
        self.global_step += 1
        
        # 计算指标
        current_metrics = self.metrics_calculator.update(outputs, targets)
        
        # 合并损失和指标
        step_metrics = {**loss_details, **current_metrics}
        
        return step_metrics
        
    def get_all_parameters(self):
        """获取所有模型参数"""
        parameters = []
        parameters.extend(self.fragile_encoder.parameters())
        parameters.extend(self.robust_encoder.parameters())
        parameters.extend(self.robust_decoder.parameters())
        parameters.extend(self.fragile_decoder.parameters())
        return parameters

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单步验证

        Args:
            batch (Dict[str, torch.Tensor]): 输入批次

        Returns:
            Dict[str, float]: 验证指标
        """
        # 设置为评估模式
        self.fragile_encoder.eval()
        self.robust_encoder.eval()
        self.robust_decoder.eval()
        self.fragile_decoder.eval()

        with torch.no_grad():
            # 前向传播
            outputs = self.forward_pass(batch)

            # 构建目标数据
            targets = {
                'original_image': batch['image_latent'].to(self.device),
                'target_message': batch['message_bits'].to(self.device),
                'target_watermark': batch['base_watermark'].to(self.device)
            }

            # 计算损失
            total_loss, loss_details = self.compute_loss(outputs, targets)

            # 计算指标
            current_metrics = self.metrics_calculator.update(outputs, targets)

            # 合并损失和指标
            step_metrics = {**loss_details, **current_metrics}

        return step_metrics

    def train_epoch(self, train_dataloader: DataLoader,
                   training_logger: TrainingLogger) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_dataloader (DataLoader): 训练数据加载器
            training_logger (TrainingLogger): 训练日志器

        Returns:
            Dict[str, float]: epoch平均指标
        """
        self.metrics_calculator.reset()
        epoch_start_time = time.time()

        # 训练循环
        for batch_idx, batch in enumerate(train_dataloader):
            try:
                # 训练步骤
                step_metrics = self.train_step(batch)

                # 记录进度
                if batch_idx % 100 == 0:  # 每100个batch记录一次
                    current_lr = self.optimizer.param_groups[0]['lr']
                    training_logger.log_batch_progress(
                        batch_idx, len(train_dataloader),
                        step_metrics, current_lr
                    )

            except Exception as e:
                logger.error(f"训练步骤 {batch_idx} 发生错误: {e}")
                continue

        # 计算epoch平均指标
        epoch_metrics = self.metrics_calculator.compute_average()
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time

        return epoch_metrics

    def validate_epoch(self, val_dataloader: DataLoader,
                      training_logger: TrainingLogger) -> Dict[str, float]:
        """
        验证一个epoch

        Args:
            val_dataloader (DataLoader): 验证数据加载器
            training_logger (TrainingLogger): 训练日志器

        Returns:
            Dict[str, float]: 验证平均指标
        """
        self.metrics_calculator.reset()

        training_logger.log_validation_start(len(val_dataloader.dataset))

        # 验证循环
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                step_metrics = self.validate_step(batch)

            except Exception as e:
                logger.error(f"验证步骤 {batch_idx} 发生错误: {e}")
                continue

        # 计算验证平均指标
        val_metrics = self.metrics_calculator.compute_average()

        training_logger.log_validation_end(val_metrics)

        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float],
                       is_best: bool = False, filename: str = None):
        """
        保存模型检查点

        Args:
            epoch (int): 当前epoch
            metrics (Dict[str, float]): 当前指标
            is_best (bool): 是否是最佳模型
            filename (str): 自定义文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"

        checkpoint_path = os.path.join(self.config['checkpoints']['save_dir'], filename)

        # 构建检查点数据
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'fragile_encoder_state_dict': self.fragile_encoder.state_dict(),
            'robust_encoder_state_dict': self.robust_encoder.state_dict(),
            'robust_decoder_state_dict': self.robust_decoder.state_dict(),
            'fragile_decoder_state_dict': self.fragile_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        # 保存检查点
        torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，额外保存单独的模型权重
        if is_best:
            models_dir = os.path.join(self.config['checkpoints']['save_dir'], 'best_models')
            os.makedirs(models_dir, exist_ok=True)

            # 保存四个模型的权重文件
            torch.save(self.fragile_encoder.state_dict(),
                      os.path.join(models_dir, 'fragile_encoder.pth'))
            torch.save(self.robust_encoder.state_dict(),
                      os.path.join(models_dir, 'robust_encoder.pth'))
            torch.save(self.robust_decoder.state_dict(),
                      os.path.join(models_dir, 'robust_decoder.pth'))
            torch.save(self.fragile_decoder.state_dict(),
                      os.path.join(models_dir, 'fragile_decoder.pth'))

        logger.info(f"检查点已保存: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点

        Args:
            checkpoint_path (str): 检查点文件路径
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态
        self.fragile_encoder.load_state_dict(checkpoint['fragile_encoder_state_dict'])
        self.robust_encoder.load_state_dict(checkpoint['robust_encoder_state_dict'])
        self.robust_decoder.load_state_dict(checkpoint['robust_decoder_state_dict'])
        self.fragile_decoder.load_state_dict(checkpoint['fragile_decoder_state_dict'])

        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复训练状态
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        logger.info(f"检查点加载完成: {checkpoint_path}")
        logger.info(f"恢复到 Epoch {self.current_epoch}, Step {self.global_step}")

        return checkpoint['metrics']

    def train(self, train_dataloader: DataLoader,
             val_dataloader: Optional[DataLoader] = None,
             training_logger: Optional[TrainingLogger] = None) -> Dict[str, float]:
        """
        主训练循环

        Args:
            train_dataloader (DataLoader): 训练数据加载器
            val_dataloader (DataLoader): 验证数据加载器
            training_logger (TrainingLogger): 训练日志器

        Returns:
            Dict[str, float]: 最佳性能指标
        """
        if training_logger is None:
            from utils.logger_config import setup_training_logger
            training_logger = setup_training_logger()

        # 记录训练开始
        training_start_time = time.time()
        training_logger.log_training_start(self.config)

        # 训练配置
        num_epochs = self.config['training']['num_epochs']
        val_frequency = self.config['training']['val_frequency']
        save_frequency = self.config['training'].get('save_frequency', 1000)

        # 性能阈值
        thresholds = self.config['checkpoints']['performance_thresholds']

        try:
            for epoch in range(self.current_epoch + 1, num_epochs + 1):
                self.current_epoch = epoch

                # 记录epoch开始
                training_logger.log_epoch_start(epoch, num_epochs)

                # 训练一个epoch
                train_metrics = self.train_epoch(train_dataloader, training_logger)

                # 验证
                val_metrics = None
                if val_dataloader and (epoch % val_frequency == 0 or epoch == num_epochs):
                    val_metrics = self.validate_epoch(val_dataloader, training_logger)

                    # 检查性能阈值
                    threshold_results = self.metrics_calculator.check_performance_thresholds(thresholds)

                    # 更新性能监控
                    is_best = self.performance_monitor.update(epoch, val_metrics)

                    if is_best:
                        training_logger.log_best_performance(epoch, val_metrics)

                        # 保存最佳模型
                        checkpoint_path = self.save_checkpoint(
                            epoch, val_metrics, is_best=True,
                            filename=f"best_model_epoch_{epoch}.pth"
                        )
                        training_logger.log_model_save(epoch, checkpoint_path, val_metrics)

                # 记录epoch结束
                training_logger.log_epoch_end(epoch, train_metrics, val_metrics)

                # 定期保存检查点
                if epoch % save_frequency == 0:
                    metrics_to_save = val_metrics if val_metrics else train_metrics
                    self.save_checkpoint(epoch, metrics_to_save)

                # 更新学习率
                if self.scheduler:
                    self.scheduler.step()

        except KeyboardInterrupt:
            training_logger.log_warning("训练被用户中断")
        except Exception as e:
            training_logger.log_error(e, "训练过程中")
            raise
        finally:
            # 训练结束
            total_time = str(timedelta(seconds=int(time.time() - training_start_time)))
            best_metrics = self.performance_monitor.get_best_metrics()
            training_logger.log_training_end(total_time, best_metrics)

        return self.performance_monitor.get_best_metrics()

    def get_model_summary(self) -> str:
        """
        获取模型摘要信息

        Returns:
            str: 格式化的模型摘要
        """
        models = {
            'FragileEncoder': self.fragile_encoder,
            'RobustEncoder': self.robust_encoder,
            'RobustDecoder': self.robust_decoder,
            'FragileDecoder': self.fragile_decoder
        }

        summary = "模型摘要:\n"
        summary += "=" * 60 + "\n"

        total_params = 0
        for name, model in models.items():
            info = model.get_model_info()
            summary += f"{name}:\n"
            summary += f"  参数量: {info['total_parameters']:,}\n"
            summary += f"  输入形状: {info['input_shape']}\n"
            summary += f"  输出形状: {info['output_shape']}\n"
            summary += "-" * 40 + "\n"
            total_params += info['total_parameters']

        summary += f"总参数量: {total_params:,}\n"
        summary += "=" * 60

        return summary
