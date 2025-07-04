"""
评估指标模块
用于计算VQ-VAE模型的各种性能指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def compute_reconstruction_quality(original_vectors: torch.Tensor, 
                                 reconstructed_vectors: torch.Tensor) -> Dict[str, float]:
    """
    计算重构质量指标
    Args:
        original_vectors: 原始向量 [N, D]
        reconstructed_vectors: 重构向量 [N, D]
    Returns:
        dict: 重构质量指标
    """
    # 确保在同一设备上
    if original_vectors.device != reconstructed_vectors.device:
        reconstructed_vectors = reconstructed_vectors.to(original_vectors.device)
    
    # 余弦相似度
    cos_sim = F.cosine_similarity(original_vectors, reconstructed_vectors, dim=-1)
    
    # MSE损失
    mse_loss = F.mse_loss(original_vectors, reconstructed_vectors, reduction='none').mean(dim=-1)
    
    # L1损失
    l1_loss = F.l1_loss(original_vectors, reconstructed_vectors, reduction='none').mean(dim=-1)
    
    # L2距离
    l2_distance = torch.norm(original_vectors - reconstructed_vectors, p=2, dim=-1)
    
    # 相对误差
    relative_error = l2_distance / (torch.norm(original_vectors, p=2, dim=-1) + 1e-8)
    
    return {
        'cosine_similarity_mean': cos_sim.mean().item(),
        'cosine_similarity_std': cos_sim.std().item(),
        'cosine_similarity_min': cos_sim.min().item(),
        'cosine_similarity_max': cos_sim.max().item(),
        'mse_loss_mean': mse_loss.mean().item(),
        'mse_loss_std': mse_loss.std().item(),
        'l1_loss_mean': l1_loss.mean().item(),
        'l2_distance_mean': l2_distance.mean().item(),
        'l2_distance_std': l2_distance.std().item(),
        'relative_error_mean': relative_error.mean().item(),
        'relative_error_std': relative_error.std().item()
    }


def compute_codebook_metrics(model, data_loader, device='cuda') -> Dict[str, float]:
    """
    计算码本相关指标
    Args:
        model: VQ-VAE模型
        data_loader: 数据加载器
        device: 计算设备
    Returns:
        dict: 码本指标
    """
    model.eval()
    all_indices = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            indices = model.encode(batch)
            all_indices.append(indices.cpu())
    
    all_indices = torch.cat(all_indices, dim=0)
    
    # 基本统计
    unique_indices = torch.unique(all_indices)
    num_used = len(unique_indices)
    total_embeddings = model.num_embeddings
    utilization = num_used / total_embeddings
    
    # 使用频率分布
    index_counts = torch.bincount(all_indices, minlength=total_embeddings)
    usage_entropy = -torch.sum(
        (index_counts / len(all_indices)) * torch.log(index_counts / len(all_indices) + 1e-8)
    ).item()
    
    # 最大使用频率
    max_usage = index_counts.max().item() / len(all_indices)
    min_usage = index_counts[index_counts > 0].min().item() / len(all_indices)
    
    return {
        'codebook_utilization': utilization,
        'num_used_embeddings': num_used,
        'total_embeddings': total_embeddings,
        'usage_entropy': usage_entropy,
        'max_usage_frequency': max_usage,
        'min_usage_frequency': min_usage,
        'usage_std': (index_counts.float() / len(all_indices)).std().item()
    }


def compute_compression_metrics(original_dim: int, 
                              compressed_bits: int, 
                              num_samples: int) -> Dict[str, float]:
    """
    计算压缩相关指标
    Args:
        original_dim: 原始向量维度
        compressed_bits: 压缩后的比特数
        num_samples: 样本数量
    Returns:
        dict: 压缩指标
    """
    # 假设原始向量使用32位浮点数
    original_bits = original_dim * 32
    compression_ratio = original_bits / compressed_bits
    space_saving = 1 - (compressed_bits / original_bits)
    
    # 总存储节省
    original_storage = num_samples * original_bits
    compressed_storage = num_samples * compressed_bits
    total_space_saved = original_storage - compressed_storage
    
    return {
        'compression_ratio': compression_ratio,
        'space_saving_ratio': space_saving,
        'original_bits_per_vector': original_bits,
        'compressed_bits_per_vector': compressed_bits,
        'total_space_saved_bits': total_space_saved,
        'total_space_saved_mb': total_space_saved / (8 * 1024 * 1024)
    }


def compute_semantic_preservation(original_vectors: torch.Tensor,
                                reconstructed_vectors: torch.Tensor,
                                k_clusters: int = 10) -> Dict[str, float]:
    """
    计算语义保持性指标
    Args:
        original_vectors: 原始向量
        reconstructed_vectors: 重构向量
        k_clusters: 聚类数量
    Returns:
        dict: 语义保持性指标
    """
    # 转换为numpy
    orig_np = original_vectors.cpu().numpy()
    recon_np = reconstructed_vectors.cpu().numpy()
    
    # 对原始向量进行聚类
    kmeans_orig = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels_orig = kmeans_orig.fit_predict(orig_np)
    
    # 使用相同的聚类中心对重构向量进行预测
    labels_recon = kmeans_orig.predict(recon_np)
    
    # 计算聚类一致性
    cluster_consistency = (labels_orig == labels_recon).mean()
    
    # 计算轮廓系数
    try:
        silhouette_orig = silhouette_score(orig_np, labels_orig)
        silhouette_recon = silhouette_score(recon_np, labels_recon)
    except:
        silhouette_orig = 0.0
        silhouette_recon = 0.0
    
    # 计算聚类中心的相似度
    centers_orig = kmeans_orig.cluster_centers_
    centers_recon = []
    for i in range(k_clusters):
        mask = labels_recon == i
        if mask.sum() > 0:
            centers_recon.append(recon_np[mask].mean(axis=0))
        else:
            centers_recon.append(np.zeros(recon_np.shape[1]))
    
    centers_recon = np.array(centers_recon)
    
    # 计算聚类中心的余弦相似度
    centers_orig_tensor = torch.from_numpy(centers_orig)
    centers_recon_tensor = torch.from_numpy(centers_recon)
    center_similarity = F.cosine_similarity(centers_orig_tensor, centers_recon_tensor, dim=1).mean().item()
    
    return {
        'cluster_consistency': cluster_consistency,
        'silhouette_score_original': silhouette_orig,
        'silhouette_score_reconstructed': silhouette_recon,
        'cluster_center_similarity': center_similarity,
        'num_clusters': k_clusters
    }


def compute_robustness_metrics(model, test_vectors: torch.Tensor, 
                             noise_levels: List[float] = [0.01, 0.05, 0.1],
                             device='cuda') -> Dict[str, Dict[str, float]]:
    """
    计算鲁棒性指标
    Args:
        model: VQ-VAE模型
        test_vectors: 测试向量
        noise_levels: 噪声水平列表
        device: 计算设备
    Returns:
        dict: 鲁棒性指标
    """
    model.eval()
    test_vectors = test_vectors.to(device)
    
    # 获取原始重构
    with torch.no_grad():
        original_recon = model(test_vectors)['x_recon']
    
    robustness_results = {}
    
    for noise_level in noise_levels:
        # 添加噪声
        noise = torch.randn_like(test_vectors) * noise_level
        noisy_vectors = test_vectors + noise
        
        # 重构噪声向量
        with torch.no_grad():
            noisy_recon = model(noisy_vectors)['x_recon']
        
        # 计算重构的稳定性
        recon_stability = F.cosine_similarity(original_recon, noisy_recon, dim=-1).mean().item()
        
        # 计算编码的稳定性
        with torch.no_grad():
            original_indices = model.encode(test_vectors)
            noisy_indices = model.encode(noisy_vectors)
        
        encoding_stability = (original_indices == noisy_indices).float().mean().item()
        
        robustness_results[f'noise_{noise_level}'] = {
            'reconstruction_stability': recon_stability,
            'encoding_stability': encoding_stability
        }
    
    return robustness_results


def comprehensive_evaluation(model, data_loader, test_vectors: torch.Tensor, 
                           device='cuda') -> Dict[str, any]:
    """
    综合评估
    Args:
        model: VQ-VAE模型
        data_loader: 数据加载器
        test_vectors: 测试向量
        device: 计算设备
    Returns:
        dict: 综合评估结果
    """
    print("Computing reconstruction quality...")
    test_vectors = test_vectors.to(device)
    with torch.no_grad():
        outputs = model(test_vectors)
        reconstructed = outputs['x_recon']
    
    reconstruction_metrics = compute_reconstruction_quality(test_vectors, reconstructed)
    
    print("Computing codebook metrics...")
    codebook_metrics = compute_codebook_metrics(model, data_loader, device)
    
    print("Computing compression metrics...")
    compression_metrics = compute_compression_metrics(
        original_dim=model.input_dim,
        compressed_bits=(model.num_embeddings - 1).bit_length(),
        num_samples=len(test_vectors)
    )
    
    print("Computing semantic preservation...")
    semantic_metrics = compute_semantic_preservation(test_vectors, reconstructed)
    
    print("Computing robustness metrics...")
    robustness_metrics = compute_robustness_metrics(model, test_vectors[:100], device=device)
    
    return {
        'reconstruction_quality': reconstruction_metrics,
        'codebook_metrics': codebook_metrics,
        'compression_metrics': compression_metrics,
        'semantic_preservation': semantic_metrics,
        'robustness_metrics': robustness_metrics,
        'model_info': {
            'input_dim': model.input_dim,
            'latent_dim': model.latent_dim,
            'num_embeddings': model.num_embeddings,
            'compression_bits': (model.num_embeddings - 1).bit_length()
        }
    }
