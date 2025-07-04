"""
语义VQ-VAE模型实现
用于将768维CLIP语义向量压缩为离散索引
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import EMAVectorQuantizer


class Encoder(nn.Module):
    """编码器：768维 -> 256维"""
    
    def __init__(self, input_dim=768, hidden_dims=[512, 384], output_dim=256, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层不加激活函数和dropout
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    """解码器：256维 -> 768维"""
    
    def __init__(self, input_dim=256, hidden_dims=[384, 512], output_dim=768, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class SemanticVQVAE(nn.Module):
    """
    语义VQ-VAE主模型
    将768维CLIP语义向量压缩为离散索引，支持精确重构
    """
    
    def __init__(self, 
                 input_dim=768,
                 latent_dim=256,
                 num_embeddings=1024,
                 commitment_cost=0.25,
                 decay=0.99,
                 dropout=0.1):
        """
        Args:
            input_dim: 输入维度（CLIP向量维度）
            latent_dim: 潜在空间维度
            num_embeddings: 码本大小
            commitment_cost: 承诺损失权重
            decay: EMA衰减系数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        # 编码器
        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            dropout=dropout
        )
        
        # 矢量量化器
        self.quantizer = EMAVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=decay
        )
        
        # 解码器
        self.decoder = Decoder(
            input_dim=latent_dim,
            output_dim=input_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        VQ-VAE前向传播 - 完整的编码-量化-解码流程

        这个函数实现了VQ-VAE的完整前向传播：
        1. 编码器将连续的CLIP向量映射到潜在空间
        2. 量化器将连续潜在向量离散化为码本索引
        3. 解码器将量化后的向量重构回原始空间
        4. 计算重构损失和量化损失

        Args:
            x: 输入语义向量 [batch_size, input_dim]
               通常是768维的CLIP语义向量
        Returns:
            dict: 包含以下键值的字典
                - x_recon: 重构向量 [batch_size, input_dim]
                - z_e: 编码器输出（量化前） [batch_size, latent_dim]
                - z_q: 量化后向量 [batch_size, latent_dim]
                - indices: 量化索引 [batch_size]，用于生成二进制串
                - recon_loss: 重构损失（MSE）
                - vq_loss: 量化损失（承诺损失）
                - total_loss: 总损失
        """
        batch_size = x.shape[0]

        # 第一步：编码 - 将768维CLIP向量压缩到256维潜在空间
        # 这一步学习语义向量的紧凑表示
        z_e = self.encoder(x)

        # 第二步：量化 - 将连续向量离散化为码本索引
        # 这是VQ-VAE的核心，实现了连续到离散的映射
        z_q, vq_loss, indices = self.quantizer(z_e)

        # 第三步：解码 - 将量化向量重构回原始768维空间
        # 解码器学习从离散表示恢复原始语义信息
        x_recon = self.decoder(z_q)

        # 计算重构损失 - 衡量重构质量
        # 使用MSE损失确保重构向量在数值上接近原始向量
        recon_loss = F.mse_loss(x_recon, x)

        # 总损失 = 重构损失 + 量化损失
        # 重构损失训练编码器和解码器，量化损失训练编码器适应码本
        total_loss = recon_loss + vq_loss

        # 记录训练统计信息（每500次前向传播记录一次）
        if self.training and hasattr(self, '_forward_count'):
            self._forward_count += 1
            if self._forward_count % 500 == 0:
                print(f"[VQ-VAE] Forward {self._forward_count}: "
                      f"recon_loss={recon_loss.item():.4f}, "
                      f"vq_loss={vq_loss.item():.4f}, "
                      f"total_loss={total_loss.item():.4f}")
        elif self.training:
            self._forward_count = 1

        return {
            'x_recon': x_recon,      # 重构的语义向量
            'total_loss': total_loss, # 总损失
            'recon_loss': recon_loss, # 重构损失
            'vq_loss': vq_loss,      # 量化损失
            'indices': indices,       # 码本索引（用于压缩）
            'z_e': z_e,              # 编码器输出（连续）
            'z_q': z_q               # 量化后向量（离散）
        }
    
    def encode(self, x):
        """编码：获取量化索引"""
        z_e = self.encoder(x)
        _, _, indices = self.quantizer(z_e)
        return indices
    
    def decode_from_indices(self, indices):
        """从索引解码：获取重构向量"""
        z_q = self.quantizer.get_codebook_entry(indices)
        x_recon = self.decoder(z_q)
        return x_recon
    
    def get_codebook_utilization(self):
        """获取码本利用率"""
        return self.quantizer.get_codebook_utilization()


def create_semantic_vqvae(config=None):
    """创建语义VQ-VAE模型的工厂函数"""
    if config is None:
        config = {
            'input_dim': 768,
            'latent_dim': 256,
            'num_embeddings': 1024,
            'commitment_cost': 0.25,
            'decay': 0.99,
            'dropout': 0.1
        }
    
    return SemanticVQVAE(**config)
