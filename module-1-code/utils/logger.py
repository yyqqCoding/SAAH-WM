"""
日志配置模块
为SAAH-WM模块一提供统一的日志记录功能
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None, 
                level: int = logging.INFO,
                console_output: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        console_output: 是否输出到控制台
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_training_logger(output_dir: str) -> logging.Logger:
    """
    获取训练专用日志记录器
    
    Args:
        output_dir: 输出目录
    
    Returns:
        logging.Logger: 训练日志记录器
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, "logs", f"training_{timestamp}.log")
    
    return setup_logger(
        name="SAAH-WM.Training",
        log_file=log_file,
        level=logging.INFO,
        console_output=True
    )


def get_evaluation_logger(output_dir: str) -> logging.Logger:
    """
    获取评估专用日志记录器
    
    Args:
        output_dir: 输出目录
    
    Returns:
        logging.Logger: 评估日志记录器
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"evaluation_{timestamp}.log")
    
    return setup_logger(
        name="SAAH-WM.Evaluation",
        log_file=log_file,
        level=logging.INFO,
        console_output=True
    )


class TrainingProgressLogger:
    """训练进度日志记录器"""
    
    def __init__(self, logger: logging.Logger, log_interval: int = 100):
        """
        初始化训练进度日志记录器
        
        Args:
            logger: 日志记录器
            log_interval: 日志记录间隔（步数）
        """
        self.logger = logger
        self.log_interval = log_interval
        self.step_count = 0
        self.epoch_start_time = None
        
    def start_epoch(self, epoch: int, total_epochs: int):
        """开始新的epoch"""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"🚀 开始训练 Epoch {epoch + 1}/{total_epochs}")
        
    def log_step(self, loss_dict: dict, lr: float, codebook_util: float):
        """记录训练步骤"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self.logger.info(
                f"Step {self.step_count}: "
                f"loss={loss_dict['total_loss']:.4f}, "
                f"recon={loss_dict['recon_loss']:.4f}, "
                f"vq={loss_dict['vq_loss']:.4f}, "
                f"lr={lr:.2e}, "
                f"util={codebook_util:.3f}"
            )
    
    def log_validation(self, val_loss: float, is_best: bool = False):
        """记录验证结果"""
        if is_best:
            self.logger.info(f"🎉 新的最佳验证损失: {val_loss:.6f}")
        else:
            self.logger.info(f"验证损失: {val_loss:.6f}")
    
    def end_epoch(self, epoch: int, avg_losses: dict):
        """结束epoch"""
        if self.epoch_start_time:
            epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
            self.logger.info(
                f"✅ Epoch {epoch + 1} 完成 "
                f"(耗时: {epoch_time:.1f}s): "
                f"avg_loss={avg_losses['total']:.4f}, "
                f"avg_recon={avg_losses['recon']:.4f}, "
                f"avg_vq={avg_losses['vq']:.4f}"
            )


class EvaluationLogger:
    """评估日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        """
        初始化评估日志记录器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        
    def log_model_info(self, model_info: dict):
        """记录模型信息"""
        self.logger.info("📊 模型信息:")
        for key, value in model_info.items():
            self.logger.info(f"   {key}: {value}")
    
    def log_test_result(self, prompt: str, metrics: dict):
        """记录单个测试结果"""
        self.logger.info(
            f"测试: '{prompt}' -> "
            f"cos_sim={metrics['cosine_similarity']:.4f}, "
            f"mse={metrics['mse_loss']:.6f}"
        )
    
    def log_summary(self, summary_metrics: dict):
        """记录汇总指标"""
        self.logger.info("📈 测试汇总:")
        for key, value in summary_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.4f}")
            else:
                self.logger.info(f"   {key}: {value}")


# 预定义的日志记录器
def get_default_logger() -> logging.Logger:
    """获取默认日志记录器"""
    return setup_logger("SAAH-WM", level=logging.INFO)


# 日志装饰器
def log_function_call(logger: logging.Logger):
    """函数调用日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"调用函数: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数 {func.__name__} 执行成功")
                return result
            except Exception as e:
                logger.error(f"函数 {func.__name__} 执行失败: {e}")
                raise
        return wrapper
    return decorator
