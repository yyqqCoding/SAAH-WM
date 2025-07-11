"""
语义哈希生成模块

实现基于CLIP模型和SimHash算法的语义哈希生成功能。
复用SEAL项目的SimHash实现，为每个prompt生成唯一的256位二进制哈希。

作者：SAAH-WM团队
"""

import torch
import numpy as np
from typing import Optional, Tuple
from transformers import CLIPModel, CLIPProcessor

from ..utils.logger_config import LoggerMixin
from ..utils.common_utils import set_random_seed, normalize_tensor


class SemanticHashGenerator(LoggerMixin):
    """
    语义哈希生成器
    
    使用CLIP模型将文本prompt编码为语义向量，
    然后通过SimHash算法生成稳定的二进制哈希。
    """
    
    def __init__(
        self, 
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        hash_bits: int = 256,
        random_seed: int = 42
    ):
        """
        初始化语义哈希生成器
        
        Args:
            clip_model: 预训练的CLIP模型
            clip_processor: CLIP处理器
            hash_bits: 哈希位数，默认256位
            random_seed: 随机种子，用于生成固定的超平面
        """
        super().__init__()
        
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.hash_bits = hash_bits
        self.random_seed = random_seed
        
        # 获取CLIP文本编码器的输出维度
        self.embedding_dim = self.clip_model.config.text_config.hidden_size
        self.log_info(f"CLIP文本编码维度: {self.embedding_dim}")
        
        # 生成固定的随机超平面用于SimHash
        self.hyperplanes = self._generate_hyperplanes()
        
        self.log_info(f"语义哈希生成器初始化完成，哈希位数: {hash_bits}")
    
    def _generate_hyperplanes(self) -> torch.Tensor:
        """
        生成用于SimHash的固定随机超平面
        
        Returns:
            形状为[hash_bits, embedding_dim]的随机超平面张量
        """
        self.log_info("正在生成SimHash超平面...")
        
        # 设置固定随机种子确保超平面的一致性
        set_random_seed(self.random_seed)
        
        # 生成随机超平面，每个超平面是一个embedding_dim维的向量
        hyperplanes = torch.randn(self.hash_bits, self.embedding_dim)
        
        # 归一化超平面向量
        hyperplanes = normalize_tensor(hyperplanes, dim=1)
        
        # 移动到与CLIP模型相同的设备
        hyperplanes = hyperplanes.to(self.clip_model.device)
        
        self.log_info(f"生成了{self.hash_bits}个超平面，维度: {self.embedding_dim}")
        return hyperplanes
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        使用CLIP模型编码文本prompt
        
        Args:
            prompt: 输入的文本prompt
            
        Returns:
            形状为[embedding_dim]的语义向量
        """
        try:
            # 使用CLIP处理器处理文本
            inputs = self.clip_processor(
                text=[prompt], 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # 移动到模型设备
            inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
            
            # 获取文本特征
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            # 归一化特征向量
            text_features = normalize_tensor(text_features, dim=1)
            
            # 返回第一个（也是唯一一个）样本的特征
            return text_features.squeeze(0)
            
        except Exception as e:
            self.log_error(f"文本编码失败: {str(e)}")
            raise
    
    def _simhash(self, embedding: torch.Tensor) -> str:
        """
        对语义向量执行SimHash算法
        
        Args:
            embedding: 形状为[embedding_dim]的语义向量
            
        Returns:
            长度为hash_bits的二进制哈希字符串
        """
        # 计算语义向量与所有超平面的点积
        # embedding: [embedding_dim], hyperplanes: [hash_bits, embedding_dim]
        # 结果: [hash_bits]
        dot_products = torch.matmul(self.hyperplanes, embedding)
        
        # 根据点积的正负生成二进制位
        # 正数为1，负数或零为0
        hash_bits = (dot_products > 0).int()
        
        # 转换为二进制字符串
        hash_string = ''.join(hash_bits.cpu().numpy().astype(str))
        
        return hash_string
    
    def generate_semantic_hash(self, prompt: str) -> str:
        """
        为给定的prompt生成语义哈希
        
        Args:
            prompt: 输入的文本prompt
            
        Returns:
            长度为hash_bits的二进制哈希字符串
        """
        self.log_info(f"开始生成语义哈希，prompt: '{prompt}'")
        
        try:
            # 步骤1：使用CLIP编码prompt为语义向量
            self.log_debug("正在编码prompt为语义向量...")
            semantic_vector = self._encode_prompt(prompt)
            self.log_debug(f"语义向量形状: {semantic_vector.shape}")
            
            # 步骤2：使用SimHash生成二进制哈希
            self.log_debug("正在执行SimHash算法...")
            hash_bits = self._simhash(semantic_vector)
            
            self.log_info(f"语义哈希生成完成，长度: {len(hash_bits)}位")
            self.log_debug(f"生成的哈希: {hash_bits[:32]}...{hash_bits[-32:]}")
            
            return hash_bits
            
        except Exception as e:
            self.log_error(f"语义哈希生成失败: {str(e)}")
            raise
    
    def verify_hash_consistency(self, prompt: str, num_tests: int = 5) -> bool:
        """
        验证同一prompt多次生成的哈希是否一致
        
        Args:
            prompt: 测试用的prompt
            num_tests: 测试次数
            
        Returns:
            是否一致
        """
        self.log_info(f"开始验证哈希一致性，测试次数: {num_tests}")
        
        hashes = []
        for i in range(num_tests):
            hash_result = self.generate_semantic_hash(prompt)
            hashes.append(hash_result)
            self.log_debug(f"第{i+1}次测试哈希: {hash_result[:16]}...")
        
        # 检查所有哈希是否相同
        is_consistent = all(h == hashes[0] for h in hashes)
        
        if is_consistent:
            self.log_info("哈希一致性验证通过")
        else:
            self.log_error("哈希一致性验证失败")
            
        return is_consistent
    
    def compare_prompts(self, prompt1: str, prompt2: str) -> Tuple[str, str, int]:
        """
        比较两个prompt的语义哈希
        
        Args:
            prompt1: 第一个prompt
            prompt2: 第二个prompt
            
        Returns:
            (hash1, hash2, hamming_distance)
        """
        self.log_info(f"比较两个prompt的语义哈希")
        self.log_debug(f"Prompt1: '{prompt1}'")
        self.log_debug(f"Prompt2: '{prompt2}'")
        
        # 生成两个哈希
        hash1 = self.generate_semantic_hash(prompt1)
        hash2 = self.generate_semantic_hash(prompt2)
        
        # 计算汉明距离
        hamming_distance = sum(b1 != b2 for b1, b2 in zip(hash1, hash2))
        
        self.log_info(f"汉明距离: {hamming_distance}/{len(hash1)} ({hamming_distance/len(hash1)*100:.1f}%)")
        
        return hash1, hash2, hamming_distance
    
    def get_hash_info(self) -> dict:
        """
        获取哈希生成器的配置信息
        
        Returns:
            配置信息字典
        """
        return {
            "hash_bits": self.hash_bits,
            "embedding_dim": self.embedding_dim,
            "random_seed": self.random_seed,
            "device": str(self.clip_model.device),
            "hyperplanes_shape": list(self.hyperplanes.shape)
        }
