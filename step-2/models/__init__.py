"""
SAAH-WM Baseline 第二步 - 网络模型模块

本模块包含四个核心网络模型：
- fragile_encoder.py: 脆弱水印编码器（IHM - Image Hiding Module）
- robust_encoder.py: 鲁棒水印编码器（BEM - Bit Embedding Module）
- robust_decoder.py: 鲁棒信息解码器（BRM - Bit Recovery Module）
- fragile_decoder.py: 脆弱水印解码器（IRM - Image Recovery Module）
"""

from .fragile_encoder import FragileEncoder
from .robust_encoder import RobustEncoder
from .robust_decoder import RobustDecoder
from .fragile_decoder import FragileDecoder

__all__ = [
    'FragileEncoder',
    'RobustEncoder', 
    'RobustDecoder',
    'FragileDecoder'
]
