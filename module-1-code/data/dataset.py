"""
语义向量数据集类
用于VQ-VAE训练的数据加载
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class SemanticVectorDataset(Dataset):
    """语义向量数据集"""
    
    def __init__(self, vectors_path, transform=None):
        """
        Args:
            vectors_path: 向量文件路径(.pt文件)
            transform: 数据变换函数
        """
        self.vectors_path = vectors_path
        self.transform = transform
        
        # 加载向量
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Vector file not found: {vectors_path}")
        
        self.vectors = torch.load(vectors_path)
        print(f"Loaded {len(self.vectors)} vectors from {vectors_path}")
        print(f"Vector shape: {self.vectors.shape}")
        
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        vector = self.vectors[idx]
        
        if self.transform:
            vector = self.transform(vector)
        
        return vector


class SemanticVectorTransform:
    """语义向量数据变换"""
    
    def __init__(self, noise_std=0.0, normalize=False):
        """
        Args:
            noise_std: 添加噪声的标准差（数据增强）
            normalize: 是否L2归一化
        """
        self.noise_std = noise_std
        self.normalize = normalize
    
    def __call__(self, vector):
        # 添加噪声（数据增强）
        if self.noise_std > 0:
            noise = torch.randn_like(vector) * self.noise_std
            vector = vector + noise
        
        # 归一化
        if self.normalize:
            vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
        
        return vector


def create_dataloaders(
    train_vectors_path,
    val_vectors_path,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    noise_std=0.0
):
    """
    创建训练和验证数据加载器
    Args:
        train_vectors_path: 训练向量文件路径
        val_vectors_path: 验证向量文件路径
        batch_size: 批处理大小
        num_workers: 数据加载进程数
        pin_memory: 是否使用pin_memory
        noise_std: 训练时添加的噪声标准差
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 创建变换
    train_transform = SemanticVectorTransform(noise_std=noise_std, normalize=True)
    val_transform = SemanticVectorTransform(noise_std=0.0, normalize=True)
    
    # 创建数据集
    train_dataset = SemanticVectorDataset(train_vectors_path, transform=train_transform)
    val_dataset = SemanticVectorDataset(val_vectors_path, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Created train loader with {len(train_dataset)} samples")
    print(f"Created val loader with {len(val_dataset)} samples")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    
    return train_loader, val_loader


def get_data_statistics(vectors_path):
    """
    获取数据统计信息
    Args:
        vectors_path: 向量文件路径
    Returns:
        dict: 统计信息
    """
    vectors = torch.load(vectors_path)
    
    stats = {
        'num_samples': vectors.shape[0],
        'vector_dim': vectors.shape[1],
        'mean': vectors.mean(dim=0),
        'std': vectors.std(dim=0),
        'min': vectors.min(dim=0)[0],
        'max': vectors.max(dim=0)[0],
        'norm_mean': torch.norm(vectors, p=2, dim=1).mean().item(),
        'norm_std': torch.norm(vectors, p=2, dim=1).std().item()
    }
    
    return stats


def visualize_data_distribution(vectors_path, save_path=None):
    """
    可视化数据分布
    Args:
        vectors_path: 向量文件路径
        save_path: 保存图片的路径
    """
    import matplotlib.pyplot as plt
    
    vectors = torch.load(vectors_path)
    
    # 计算向量的L2范数
    norms = torch.norm(vectors, p=2, dim=1).numpy()
    
    # 绘制分布图
    plt.figure(figsize=(12, 4))
    
    # 范数分布
    plt.subplot(1, 3, 1)
    plt.hist(norms, bins=50, alpha=0.7)
    plt.title('Vector L2 Norm Distribution')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    
    # 第一个维度的分布
    plt.subplot(1, 3, 2)
    plt.hist(vectors[:, 0].numpy(), bins=50, alpha=0.7)
    plt.title('First Dimension Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # 向量间余弦相似度分布（采样1000个向量对）
    plt.subplot(1, 3, 3)
    if len(vectors) > 1000:
        indices = torch.randperm(len(vectors))[:1000]
        sample_vectors = vectors[indices]
    else:
        sample_vectors = vectors
    
    # 计算余弦相似度矩阵
    normalized_vectors = torch.nn.functional.normalize(sample_vectors, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t())
    
    # 提取上三角部分（排除对角线）
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    similarities = similarity_matrix[mask].numpy()
    
    plt.hist(similarities, bins=50, alpha=0.7)
    plt.title('Pairwise Cosine Similarity')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 测试数据集
    vectors_path = "./data/processed/train_vectors.pt"
    
    if os.path.exists(vectors_path):
        # 获取统计信息
        stats = get_data_statistics(vectors_path)
        print("Data statistics:")
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    print(f"  {key}: {value.item():.4f}")
                else:
                    print(f"  {key}: shape {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # 可视化数据分布
        visualize_data_distribution(vectors_path, "./data/processed/data_distribution.png")
    else:
        print(f"Vector file not found: {vectors_path}")
        print("Please run preprocess_data.py first")
