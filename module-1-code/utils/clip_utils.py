"""
CLIP相关工具函数
用于文本编码和语义向量提取
"""

import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Union
import numpy as np


class CLIPTextEncoder:
    """CLIP文本编码器封装类"""
    
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        """
        初始化CLIP文本编码器
        Args:
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.device = device
        self.model_name = model_name
        
        # 加载tokenizer和模型
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded CLIP model: {model_name}")
        print(f"Text encoder output dimension: {self.model.config.hidden_size}")
    
    def encode_text(self, texts: Union[str, List[str]], normalize=True):
        """
        编码文本为CLIP语义向量 - SAAH-WM系统的入口点

        这个函数是整个SAAH-WM系统的关键组件，将自然语言文本
        转换为768维的语义向量，为后续的VQ-VAE压缩做准备。

        处理流程：
        1. 文本预处理：tokenization, padding, truncation
        2. 通过CLIP文本编码器提取语义特征
        3. L2归一化确保向量在单位球面上
        4. 返回标准化的语义向量

        Args:
            texts: 单个文本字符串或文本列表
                  例如: "a cat sitting on a chair" 或
                       ["a cat", "a dog", "a bird"]
            normalize: 是否进行L2归一化 (默认True)
                      归一化确保所有向量具有相同的模长
        Returns:
            torch.Tensor: 语义向量 [batch_size, 768]
                         包含文本语义信息的高维向量表示
        """
        # 统一处理：将单个文本转换为列表格式
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        # 文本预处理：tokenization + padding + truncation
        # max_length=77 是CLIP模型的标准输入长度限制
        inputs = self.tokenizer(
            texts,
            padding=True,        # 填充到批次中最长序列的长度
            truncation=True,     # 截断超长序列
            max_length=77,       # CLIP的最大token长度
            return_tensors="pt"  # 返回PyTorch张量
        )

        # 将输入张量移动到指定设备（GPU/CPU）
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播提取文本语义特征
        with torch.no_grad():  # 推理模式，不计算梯度，节省内存
            # 通过CLIP文本编码器获取语义表示
            outputs = self.model(**inputs)

            # 使用pooled_output作为文本的全局语义表示
            # 这是经过池化的768维向量，包含了整个文本的语义信息
            text_embeddings = outputs.pooler_output

        # L2归一化：将向量投影到单位球面
        # 这确保了所有向量具有相同的模长，便于后续的距离计算和量化
        if normalize:
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # 如果输入是单个文本，返回单个向量（去除批次维度）
        if single_text:
            return text_embeddings.squeeze(0)

        return text_embeddings
    
    def encode_batch(self, texts: List[str], batch_size=32, normalize=True):
        """
        批量编码文本
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否归一化
        Returns:
            torch.Tensor: 所有文本的语义向量
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts, normalize=normalize)
            all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)


def compute_cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    Args:
        vec1, vec2: 输入向量
    Returns:
        float: 余弦相似度
    """
    return F.cosine_similarity(vec1, vec2, dim=-1).mean().item()


def compute_reconstruction_metrics(original_vectors, reconstructed_vectors):
    """
    计算重构质量指标
    Args:
        original_vectors: 原始向量
        reconstructed_vectors: 重构向量
    Returns:
        dict: 包含各种指标的字典
    """
    # 余弦相似度
    cos_sim = compute_cosine_similarity(original_vectors, reconstructed_vectors)
    
    # MSE损失
    mse_loss = F.mse_loss(original_vectors, reconstructed_vectors).item()
    
    # L2距离
    l2_distance = torch.norm(original_vectors - reconstructed_vectors, p=2, dim=-1).mean().item()
    
    return {
        'cosine_similarity': cos_sim,
        'mse_loss': mse_loss,
        'l2_distance': l2_distance
    }


def prompt_to_bits(prompt: str, vqvae_model, clip_encoder, num_bits=10):
    """
    将文本提示转换为二进制串
    Args:
        prompt: 文本提示
        vqvae_model: 训练好的VQ-VAE模型
        clip_encoder: CLIP编码器
        num_bits: 索引的比特数
    Returns:
        str: 二进制串
    """
    # 获取CLIP向量
    clip_vector = clip_encoder.encode_text(prompt)
    
    # 获取VQ-VAE索引
    with torch.no_grad():
        indices = vqvae_model.encode(clip_vector)
    
    # 转换为二进制
    index = indices[0].item()  # 假设batch_size=1
    binary_str = format(index, f'0{num_bits}b')
    
    return binary_str


def bits_to_reconstructed_vector(bits: str, vqvae_model):
    """
    将二进制串转换回重构的语义向量
    Args:
        bits: 二进制串
        vqvae_model: 训练好的VQ-VAE模型
    Returns:
        torch.Tensor: 重构的语义向量
    """
    # 转换为索引
    index = int(bits, 2)
    indices = torch.tensor([index], device=next(vqvae_model.parameters()).device)
    
    # 重构向量
    with torch.no_grad():
        reconstructed = vqvae_model.decode_from_indices(indices)
    
    return reconstructed


def test_compression_pipeline(prompts: List[str], vqvae_model, clip_encoder, num_bits=10):
    """
    测试完整的压缩-重构流水线
    Args:
        prompts: 测试文本列表
        vqvae_model: VQ-VAE模型
        clip_encoder: CLIP编码器
        num_bits: 索引比特数
    Returns:
        dict: 测试结果
    """
    results = []
    
    for prompt in prompts:
        # 原始向量
        original_vector = clip_encoder.encode_text(prompt)
        
        # 压缩为比特串
        bits = prompt_to_bits(prompt, vqvae_model, clip_encoder, num_bits)
        
        # 重构向量
        reconstructed_vector = bits_to_reconstructed_vector(bits, vqvae_model)
        
        # 计算指标
        metrics = compute_reconstruction_metrics(original_vector, reconstructed_vector)
        
        results.append({
            'prompt': prompt,
            'bits': bits,
            'metrics': metrics
        })
    
    return results
