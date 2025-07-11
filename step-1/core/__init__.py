"""
SAAH-WM Baseline 第一步核心模块

本包包含SAAH-WM基线实现的四个核心模块：
1. semantic_hash - 语义哈希生成
2. base_watermark - 基准水印生成  
3. attention_extractor - 注意力图谱提取与掩码生成
4. message_packet - 信息包生成

作者：SAAH-WM团队
版本：1.0.0
"""

from .semantic_hash import SemanticHashGenerator
from .base_watermark import BaseWatermarkGenerator
from .attention_extractor import AttentionExtractor, AttentionStore
from .message_packet import MessagePacketGenerator

__all__ = [
    'SemanticHashGenerator',
    'BaseWatermarkGenerator', 
    'AttentionExtractor',
    'AttentionStore',
    'MessagePacketGenerator'
]

__version__ = "1.0.0"
