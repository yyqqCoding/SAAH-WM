"""
SAAH-WM Baseline 第二步 - 日志配置模块

本模块提供完整的日志系统配置，支持：
- 文件日志和控制台日志
- 不同级别的日志记录
- 训练进度监控
- 错误异常追踪
- 中文友好的日志格式
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# 尝试导入colorlog，如果没有则使用标准日志
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logger(name: str = 'saah_wm', 
                log_dir: str = 'logs',
                log_level: str = 'INFO',
                console_output: bool = True,
                file_output: bool = True,
                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        name (str): 日志器名称，默认'saah_wm'
        log_dir (str): 日志文件目录，默认'logs'
        log_level (str): 日志级别，默认'INFO'
        console_output (bool): 是否输出到控制台，默认True
        file_output (bool): 是否输出到文件，默认True
        max_file_size (int): 单个日志文件最大大小（字节），默认10MB
        backup_count (int): 保留的日志文件数量，默认5
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建日志目录
    if file_output and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 日志格式
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 控制台处理器（带颜色）
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 彩色格式（如果可用）
        if HAS_COLORLOG:
            color_formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s%(asctime)s | %(levelname)s | %(message)s%(reset)s',
                datefmt='%H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
        else:
            # 使用简单格式
            console_handler.setFormatter(simple_formatter)

        logger.addHandler(console_handler)
    
    # 文件处理器
    if file_output:
        # 主日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # 错误日志文件
        error_filename = os.path.join(log_dir, f'error_{timestamp}.log')
        error_handler = logging.FileHandler(error_filename, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    # 记录初始化信息
    logger.info(f"日志系统初始化完成")
    logger.info(f"日志级别: {log_level}")
    logger.info(f"控制台输出: {console_output}")
    logger.info(f"文件输出: {file_output}")
    if file_output:
        logger.info(f"日志目录: {log_dir}")
        logger.info(f"主日志文件: {log_filename}")
        logger.info(f"错误日志文件: {error_filename}")
    
    return logger


class TrainingLogger:
    """
    训练专用日志器
    
    提供训练过程中的专门日志记录功能，包括：
    - 训练进度记录
    - 损失和指标记录
    - 模型状态记录
    - 错误和异常处理
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_start_time = None
        self.batch_start_time = None
        
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 SAAH-WM Baseline 第二步训练开始")
        self.logger.info("=" * 80)
        
        # 记录配置信息
        self.logger.info("训练配置:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
                
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.epoch_start_time = datetime.now()
        self.logger.info("-" * 60)
        self.logger.info(f"📈 Epoch {epoch}/{total_epochs} 开始")
        self.logger.info(f"开始时间: {self.epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def log_batch_progress(self, batch_idx: int, total_batches: int, 
                          loss_details: Dict[str, float], 
                          learning_rate: float,
                          log_interval: int = 100):
        """记录批次进度"""
        if batch_idx % log_interval == 0:
            progress = batch_idx / total_batches * 100
            
            # 构建损失信息字符串
            loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in loss_details.items()])
            
            self.logger.info(
                f"Batch {batch_idx:4d}/{total_batches} ({progress:5.1f}%) | "
                f"LR: {learning_rate:.2e} | {loss_str}"
            )
            
    def log_epoch_end(self, epoch: int, train_metrics: Dict[str, float], 
                     val_metrics: Optional[Dict[str, float]] = None):
        """记录epoch结束"""
        if self.epoch_start_time:
            epoch_time = datetime.now() - self.epoch_start_time
            self.logger.info(f"⏱️  Epoch {epoch} 耗时: {epoch_time}")
        
        # 记录训练指标
        self.logger.info("📊 训练指标:")
        for key, value in train_metrics.items():
            if 'accuracy' in key or 'rate' in key:
                self.logger.info(f"  {key}: {value:.4f}")
            elif 'loss' in key:
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value:.2f}")
        
        # 记录验证指标
        if val_metrics:
            self.logger.info("🔍 验证指标:")
            for key, value in val_metrics.items():
                if 'accuracy' in key or 'rate' in key:
                    self.logger.info(f"  {key}: {value:.4f}")
                elif 'loss' in key:
                    self.logger.info(f"  {key}: {value:.6f}")
                else:
                    self.logger.info(f"  {key}: {value:.2f}")
                    
    def log_model_save(self, epoch: int, filepath: str, metrics: Dict[str, float]):
        """记录模型保存"""
        self.logger.info(f"💾 模型已保存 (Epoch {epoch})")
        self.logger.info(f"保存路径: {filepath}")
        self.logger.info(f"性能指标: {metrics}")
        
    def log_best_performance(self, epoch: int, metrics: Dict[str, float]):
        """记录最佳性能"""
        self.logger.info("🏆 发现新的最佳性能!")
        self.logger.info(f"Epoch: {epoch}")
        for key, value in metrics.items():
            if 'accuracy' in key or 'rate' in key:
                self.logger.info(f"  {key}: {value:.4f}")
            elif 'loss' in key:
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value:.2f}")
                
    def log_validation_start(self, num_samples: int):
        """记录验证开始"""
        self.logger.info(f"🔍 开始验证 (样本数: {num_samples})")
        
    def log_validation_end(self, metrics: Dict[str, float]):
        """记录验证结束"""
        self.logger.info("✅ 验证完成")
        
        # 检查性能阈值
        thresholds = {
            'psnr': 38.0,
            'ssim': 0.95,
            'bit_accuracy': 0.995
        }
        
        self.logger.info("🎯 性能阈值检查:")
        for metric, threshold in thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                status = "✅ 达标" if value >= threshold else "❌ 未达标"
                self.logger.info(f"  {metric}: {value:.4f} (阈值: {threshold}) {status}")
                
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        self.logger.error(f"❌ 发生错误 {context}")
        self.logger.error(f"错误类型: {type(error).__name__}")
        self.logger.error(f"错误信息: {str(error)}")
        self.logger.exception("详细错误堆栈:")
        
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(f"⚠️  {message}")
        
    def log_training_end(self, total_time: str, best_metrics: Dict[str, float]):
        """记录训练结束"""
        self.logger.info("=" * 80)
        self.logger.info("🎉 训练完成!")
        self.logger.info(f"总耗时: {total_time}")
        self.logger.info("最佳性能:")
        for key, value in best_metrics.items():
            if 'accuracy' in key or 'rate' in key:
                self.logger.info(f"  {key}: {value:.4f}")
            elif 'loss' in key:
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value:.2f}")
        self.logger.info("=" * 80)


def setup_training_logger(log_dir: str = 'logs', 
                         log_level: str = 'INFO') -> TrainingLogger:
    """
    设置训练专用日志器
    
    Args:
        log_dir (str): 日志目录
        log_level (str): 日志级别
        
    Returns:
        TrainingLogger: 训练日志器
    """
    base_logger = setup_logger(
        name='saah_wm_training',
        log_dir=log_dir,
        log_level=log_level,
        console_output=True,
        file_output=True
    )
    
    return TrainingLogger(base_logger)


if __name__ == "__main__":
    # 测试日志系统
    
    # 基础日志器测试
    print("测试基础日志器:")
    logger = setup_logger(log_level='DEBUG')
    
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    # 训练日志器测试
    print("\n测试训练日志器:")
    training_logger = setup_training_logger()
    
    # 模拟训练配置
    config = {
        'model': {
            'latent_channels': 4,
            'hidden_dim': 256
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 100
        }
    }
    
    # 模拟训练过程
    training_logger.log_training_start(config)
    
    training_logger.log_epoch_start(1, 100)
    
    # 模拟批次进度
    loss_details = {
        'total_loss': 0.1234,
        'image_loss': 0.0456,
        'robust_loss': 0.0678,
        'fragile_loss': 0.0100
    }
    training_logger.log_batch_progress(100, 1000, loss_details, 1e-4)
    
    # 模拟epoch结束
    train_metrics = {
        'psnr': 39.5,
        'ssim': 0.96,
        'bit_accuracy': 0.997,
        'total_loss': 0.0987
    }
    
    val_metrics = {
        'psnr': 38.2,
        'ssim': 0.95,
        'bit_accuracy': 0.995,
        'total_loss': 0.1123
    }
    
    training_logger.log_epoch_end(1, train_metrics, val_metrics)
    
    # 模拟验证过程
    training_logger.log_validation_start(1000)
    training_logger.log_validation_end(val_metrics)
    
    # 模拟最佳性能
    training_logger.log_best_performance(1, val_metrics)
    
    # 模拟模型保存
    training_logger.log_model_save(1, "checkpoints/best_model.pth", val_metrics)
    
    # 模拟错误
    try:
        raise ValueError("这是一个测试错误")
    except Exception as e:
        training_logger.log_error(e, "在测试过程中")
    
    # 模拟警告
    training_logger.log_warning("这是一个测试警告")
    
    # 模拟训练结束
    training_logger.log_training_end("2小时30分钟", val_metrics)
    
    print("\n日志测试完成! 请检查logs目录中的日志文件。")
