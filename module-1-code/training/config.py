"""
训练配置文件
包含所有超参数和训练设置
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    VQ-VAE模型配置类

    定义了语义VQ-VAE模型的所有关键超参数。这些参数直接影响
    模型的压缩性能、重构质量和训练稳定性。
    """
    # 输入输出维度配置
    input_dim: int = 768          # CLIP语义向量维度（固定为768）
    latent_dim: int = 256         # 潜在空间维度，编码器输出维度

    # 量化配置 - 决定压缩性能
    num_embeddings: int = 1024    # 码本大小，决定压缩比特数 log2(1024)=10bits
    commitment_cost: float = 0.25 # 承诺损失权重，控制编码器适应码本的程度
    decay: float = 0.99           # EMA衰减系数，控制码本更新平滑程度

    # 正则化配置
    dropout: float = 0.1          # Dropout概率，防止过拟合


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据相关
    data_dir: str = "./data/processed"
    train_vectors_file: str = "train_vectors.pt"
    val_vectors_file: str = "val_vectors.pt"
    
    # 训练超参数
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    noise_std: float = 0.01  # 训练时的噪声增强
    
    # 优化器
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # 设备和并行
    device: str = "cuda"
    mixed_precision: bool = True
    
    # 日志和保存
    log_interval: int = 100  # 每多少步打印一次
    val_interval: int = 1000  # 每多少步验证一次
    save_interval: int = 5000  # 每多少步保存一次
    
    # 输出目录
    output_dir: str = "./outputs"
    experiment_name: str = "semantic_vqvae"
    
    # 早停
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 恢复训练
    resume_from_checkpoint: Optional[str] = None


@dataclass
class Config:
    """完整配置"""
    model: ModelConfig
    training: TrainingConfig
    
    def __post_init__(self):
        # 创建输出目录
        self.training.output_dir = os.path.join(
            self.training.output_dir, 
            self.training.experiment_name
        )
        os.makedirs(self.training.output_dir, exist_ok=True)
        
        # 设置数据路径
        self.training.train_vectors_path = os.path.join(
            self.training.data_dir, 
            self.training.train_vectors_file
        )
        self.training.val_vectors_path = os.path.join(
            self.training.data_dir, 
            self.training.val_vectors_file
        )


def get_default_config():
    """获取默认配置"""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    return Config(model=model_config, training=training_config)


def get_small_config():
    """获取小规模测试配置"""
    model_config = ModelConfig(
        num_embeddings=256,  # 更小的码本
        latent_dim=128,      # 更小的潜在维度
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        num_epochs=20,
        learning_rate=2e-3,
    )
    
    return Config(model=model_config, training=training_config)


def get_large_config():
    """获取大规模配置"""
    model_config = ModelConfig(
        num_embeddings=4096,  # 更大的码本
        latent_dim=512,       # 更大的潜在维度
        dropout=0.05,         # 更小的dropout
    )
    
    training_config = TrainingConfig(
        batch_size=256,
        num_epochs=200,
        learning_rate=5e-4,
        warmup_epochs=10,
    )
    
    return Config(model=model_config, training=training_config)


def save_config(config: Config, save_path: str):
    """保存配置到文件"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Config saved to {save_path}")


def load_config(config_path: str) -> Config:
    """从文件加载配置"""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    model_config = ModelConfig(**config_dict['model'])
    training_config = TrainingConfig(**config_dict['training'])
    
    return Config(model=model_config, training=training_config)
