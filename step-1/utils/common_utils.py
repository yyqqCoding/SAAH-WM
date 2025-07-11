"""
通用工具函数模块

提供项目中常用的工具函数和辅助方法。

作者：SAAH-WM团队
"""

import torch
import numpy as np
import random
from typing import Union, List, Tuple
import os


def set_random_seed(seed: int = 42):
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(dir_path: str):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将PyTorch张量转换为NumPy数组
    
    Args:
        tensor: 输入张量
        
    Returns:
        NumPy数组
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.numpy()


def numpy_to_tensor(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    将NumPy数组转换为PyTorch张量
    
    Args:
        array: 输入数组
        device: 目标设备
        
    Returns:
        PyTorch张量
    """
    tensor = torch.from_numpy(array)
    return tensor.to(device)


def bits_to_int(bits: str) -> int:
    """
    将二进制字符串转换为整数
    
    Args:
        bits: 二进制字符串
        
    Returns:
        整数值
    """
    return int(bits, 2)


def int_to_bits(value: int, length: int) -> str:
    """
    将整数转换为指定长度的二进制字符串
    
    Args:
        value: 整数值
        length: 二进制字符串长度
        
    Returns:
        二进制字符串
    """
    return format(value, f'0{length}b')


def string_to_bits(text: str, encoding: str = 'utf-8') -> str:
    """
    将字符串转换为二进制字符串
    
    Args:
        text: 输入字符串
        encoding: 字符编码
        
    Returns:
        二进制字符串
    """
    # 将字符串编码为字节
    byte_data = text.encode(encoding)
    
    # 将每个字节转换为8位二进制
    bits = ''.join(format(byte, '08b') for byte in byte_data)
    
    return bits


def bits_to_string(bits: str, encoding: str = 'utf-8') -> str:
    """
    将二进制字符串转换为字符串
    
    Args:
        bits: 二进制字符串
        encoding: 字符编码
        
    Returns:
        解码后的字符串
    """
    # 确保位数是8的倍数
    if len(bits) % 8 != 0:
        raise ValueError(f"二进制字符串长度必须是8的倍数，当前长度: {len(bits)}")
    
    # 将二进制字符串分割为8位一组
    byte_chunks = [bits[i:i+8] for i in range(0, len(bits), 8)]
    
    # 将每组转换为字节
    byte_data = bytes(int(chunk, 2) for chunk in byte_chunks)
    
    # 解码为字符串
    return byte_data.decode(encoding)


def calculate_hamming_distance(bits1: str, bits2: str) -> int:
    """
    计算两个二进制字符串的汉明距离
    
    Args:
        bits1: 第一个二进制字符串
        bits2: 第二个二进制字符串
        
    Returns:
        汉明距离
    """
    if len(bits1) != len(bits2):
        raise ValueError("两个二进制字符串长度必须相同")
    
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))


def normalize_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    对张量进行L2归一化
    
    Args:
        tensor: 输入张量
        dim: 归一化维度
        
    Returns:
        归一化后的张量
    """
    return torch.nn.functional.normalize(tensor, p=2, dim=dim)


def pad_bits(bits: str, target_length: int, pad_char: str = '0') -> str:
    """
    填充二进制字符串到指定长度
    
    Args:
        bits: 输入二进制字符串
        target_length: 目标长度
        pad_char: 填充字符
        
    Returns:
        填充后的二进制字符串
    """
    if len(bits) >= target_length:
        return bits[:target_length]
    
    padding_length = target_length - len(bits)
    return bits + pad_char * padding_length


def validate_bits(bits: str) -> bool:
    """
    验证字符串是否为有效的二进制字符串
    
    Args:
        bits: 待验证的字符串
        
    Returns:
        是否为有效的二进制字符串
    """
    return all(c in '01' for c in bits)


def get_memory_usage() -> dict:
    """
    获取当前内存使用情况
    
    Returns:
        内存使用信息字典
    """
    import psutil
    
    # 系统内存信息
    memory = psutil.virtual_memory()
    
    info = {
        "system_memory_total": memory.total,
        "system_memory_available": memory.available,
        "system_memory_percent": memory.percent,
        "system_memory_used": memory.used
    }
    
    # GPU内存信息（如果可用）
    if torch.cuda.is_available():
        info.update({
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_reserved": torch.cuda.memory_reserved(),
            "gpu_memory_max_allocated": torch.cuda.max_memory_allocated(),
            "gpu_memory_max_reserved": torch.cuda.max_memory_reserved()
        })
    
    return info
