"""
SAAH-WM Baseline 第二步 - 核心训练模块

本模块包含训练系统的核心组件：
- trainer.py: 主训练器，负责整个训练流程的控制和协调
"""

from .trainer import WatermarkTrainer

__all__ = ['WatermarkTrainer']
