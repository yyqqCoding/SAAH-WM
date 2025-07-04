"""
优化的矢量量化器实现
基于VQ-VAE-2的EMA更新机制，适配语义向量压缩任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """
    使用指数移动平均(EMA)更新的矢量量化器
    相比标准VQ-VAE，EMA方法更稳定，收敛更快
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        """
        Args:
            num_embeddings: 码本大小，即标准向量的数量
            embedding_dim: 每个标准向量的维度
            commitment_cost: 承诺损失的权重系数
            decay: EMA衰减系数
            epsilon: 数值稳定性参数
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 初始化码本 - 使用正态分布初始化
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())
        
    def forward(self, inputs):
        """
        前向传播 - VQ-VAE的核心量化过程

        这个函数实现了矢量量化的核心逻辑：
        1. 计算输入向量与码本中所有向量的距离
        2. 找到最近的码本向量作为量化结果
        3. 使用EMA更新码本（仅在训练时）
        4. 计算承诺损失并应用straight-through estimator

        Args:
            inputs: 输入张量 [batch_size, embedding_dim]
                   通常是编码器的输出，需要被量化为离散表示
        Returns:
            quantized: 量化后的张量，形状与输入相同
            loss: VQ损失（承诺损失），用于训练编码器
            encoding_indices: 编码索引，用于生成二进制串
        """
        # 保存原始形状，用于最后恢复
        input_shape = inputs.shape
        batch_size = input_shape[0]

        # 将输入展平为 [N, embedding_dim] 以便批量计算距离
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算输入向量与码本中每个向量的欧氏距离
        # 使用数学恒等式: ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x*e
        # 这比直接计算距离更高效，避免了大量的广播操作
        input_norm = torch.sum(flat_input**2, dim=1, keepdim=True)  # [N, 1]
        embed_norm = torch.sum(self.embed**2, dim=0, keepdim=True)  # [1, num_embeddings]
        dot_product = torch.matmul(flat_input, self.embed)          # [N, num_embeddings]

        distances = input_norm + embed_norm - 2 * dot_product       # [N, num_embeddings]

        # 找到距离最小的码本向量索引
        # argmin返回每行最小值的索引，即每个输入向量对应的最佳码本索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [N, 1]

        # 创建one-hot编码矩阵，用于后续的码本更新和量化
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)  # [N, num_embeddings]

        # 执行量化：用选中的码本向量替换原始输入
        # 这里使用矩阵乘法实现批量查找，等价于 self.embed[:, encoding_indices]
        quantized = torch.matmul(encodings, self.embed.t()).view(input_shape)

        # 在训练模式下更新码本（使用EMA）
        if self.training:
            self._update_codebook(flat_input, encodings)

            # 记录量化统计信息（用于调试）
            unique_indices = torch.unique(encoding_indices)
            utilization = len(unique_indices) / self.num_embeddings
            if hasattr(self, '_step_count'):
                self._step_count += 1
                if self._step_count % 1000 == 0:  # 每1000步记录一次
                    print(f"[VQ] Step {self._step_count}: Codebook utilization = {utilization:.3f}")
            else:
                self._step_count = 1

        # 计算承诺损失 (commitment loss)
        # 这个损失鼓励编码器的输出接近选中的码本向量
        # 使用detach()确保梯度只流向编码器，不影响码本
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator: 前向传播使用量化值，反向传播使用原始梯度
        # 这是VQ-VAE的关键技巧，允许梯度穿过离散的量化操作
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices.squeeze()
    
    def _update_codebook(self, flat_input, encodings):
        """使用EMA更新码本"""
        # 更新聚类大小
        self.cluster_size.data.mul_(self.decay).add_(
            torch.sum(encodings, 0), alpha=1 - self.decay)
        
        # 更新嵌入向量的移动平均
        dw = torch.matmul(encodings.t(), flat_input)
        self.embed_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # 归一化
        n = torch.sum(self.cluster_size.data)
        cluster_size = (
            (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        self.embed.data.copy_(embed_normalized)
    
    def get_codebook_entry(self, indices):
        """根据索引获取码本向量"""
        return F.embedding(indices, self.embed.t())
    
    def get_codebook_utilization(self):
        """计算码本利用率"""
        return (self.cluster_size > 0).float().mean().item()
