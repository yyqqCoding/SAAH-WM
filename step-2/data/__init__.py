"""
SAAH-WM Baseline 第二步 - 数据加载模块

本模块包含数据加载和预处理组件：
- coco_dataloader.py: COCO2017数据集加载器，集成第一步的四个核心模块
"""

from .coco_dataloader import COCOWatermarkDataset, create_dataloader

__all__ = ['COCOWatermarkDataset', 'create_dataloader']
