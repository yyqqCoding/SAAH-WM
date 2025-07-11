"""
日志配置模块

提供统一的日志配置功能，支持文件和控制台输出，
包含详细的模型加载状态、处理进度、错误异常处理等信息。

作者：SAAH-WM团队
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "SAAH-WM", 
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置并配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不写入文件
        console_output: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器，避免重复
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_logger() -> logging.Logger:
    """
    获取默认配置的日志记录器
    
    Returns:
        默认日志记录器
    """
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/saah_wm_{timestamp}.log"
    
    return setup_logger(
        name="SAAH-WM",
        log_level="INFO", 
        log_file=log_file,
        console_output=True
    )


class LoggerMixin:
    """
    日志记录器混入类
    
    为其他类提供统一的日志记录功能
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_default_logger()
    
    def log_info(self, message: str):
        """记录信息级别日志"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告级别日志"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """记录错误级别日志"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """记录调试级别日志"""
        self.logger.debug(message)
    
    def log_progress(self, current: int, total: int, task_name: str = "处理"):
        """记录进度信息"""
        percentage = (current / total) * 100
        self.logger.info(f"{task_name}进度: {current}/{total} ({percentage:.1f}%)")
