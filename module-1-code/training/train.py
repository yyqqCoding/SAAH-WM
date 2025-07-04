"""
VQ-VAE训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae_model import SemanticVQVAE
from data.dataset import create_dataloaders
from training.config import get_default_config, get_small_config, get_large_config, save_config


class Trainer:
    """VQ-VAE训练器"""
    
    def __init__(self, config):
        """
        初始化训练器

        设置模型、数据加载器、优化器等训练所需的所有组件
        """
        self.config = config
        self.device = torch.device(config.training.device)

        print("=" * 60)
        print("🚀 初始化SAAH-WM模块一训练器")
        print("=" * 60)

        # 创建输出目录
        os.makedirs(config.training.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.training.output_dir, "logs"), exist_ok=True)

        # 创建模型
        print("📦 创建VQ-VAE模型...")
        self.model = SemanticVQVAE(
            input_dim=config.model.input_dim,
            latent_dim=config.model.latent_dim,
            num_embeddings=config.model.num_embeddings,
            commitment_cost=config.model.commitment_cost,
            decay=config.model.decay,
            dropout=config.model.dropout
        ).to(self.device)

        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✓ 模型参数数量: {model_params:,}")
        print(f"   ✓ 输入维度: {config.model.input_dim}")
        print(f"   ✓ 潜在维度: {config.model.latent_dim}")
        print(f"   ✓ 码本大小: {config.model.num_embeddings}")
        print(f"   ✓ 压缩比特数: {(config.model.num_embeddings - 1).bit_length()}")

        # 创建数据加载器
        print("📊 创建数据加载器...")
        try:
            self.train_loader, self.val_loader = create_dataloaders(
                train_vectors_path=config.training.train_vectors_path,
                val_vectors_path=config.training.val_vectors_path,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                noise_std=config.training.noise_std
            )
            print(f"   ✓ 训练样本数: {len(self.train_loader.dataset):,}")
            print(f"   ✓ 验证样本数: {len(self.val_loader.dataset):,}")
            print(f"   ✓ 批处理大小: {config.training.batch_size}")
            print(f"   ✓ 训练批次数: {len(self.train_loader)}")
        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            print("   💡 请确保已运行数据预处理脚本")
            raise

        # 创建优化器
        print("⚙️ 创建优化器和调度器...")
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        print(f"   ✓ 优化器: {config.training.optimizer}")
        print(f"   ✓ 学习率: {config.training.learning_rate}")
        print(f"   ✓ 权重衰减: {config.training.weight_decay}")

        # 混合精度训练
        self.scaler = GradScaler() if config.training.mixed_precision else None
        if config.training.mixed_precision:
            print("   ✓ 启用混合精度训练")

        # 日志记录
        self.writer = SummaryWriter(log_dir=os.path.join(config.training.output_dir, "logs"))
        print(f"   ✓ TensorBoard日志: {os.path.join(config.training.output_dir, 'logs')}")

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        print(f"🖥️ 训练设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"   ✓ GPU型号: {torch.cuda.get_device_name()}")
            print(f"   ✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        print("=" * 60)
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.training.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self):
        """
        训练一个epoch

        执行完整的训练循环，包括前向传播、反向传播、参数更新
        以及定期的验证和检查点保存
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0

        # 创建进度条，显示训练进度和关键指标
        pbar = tqdm(self.train_loader, desc=f"🔄 Epoch {self.epoch + 1}/{self.config.training.num_epochs}")

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # 前向传播 - 支持混合精度训练
            if self.scaler:
                with autocast():
                    outputs = self.model(batch)
                    loss = outputs['total_loss']

                # 反向传播 - 混合精度
                self.scaler.scale(loss).backward()

                # 梯度裁剪（防止梯度爆炸）
                if hasattr(self.config.training, 'max_grad_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = outputs['total_loss']

                # 反向传播 - 标准精度
                loss.backward()

                # 梯度裁剪
                if hasattr(self.config.training, 'max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

                self.optimizer.step()

            # 累计损失统计
            total_loss += loss.item()
            total_recon_loss += outputs['recon_loss'].item()
            total_vq_loss += outputs['vq_loss'].item()

            # 获取码本利用率
            codebook_util = self.model.get_codebook_utilization()

            # 更新进度条显示
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{outputs['recon_loss'].item():.4f}",
                'vq': f"{outputs['vq_loss'].item():.4f}",
                'util': f"{codebook_util:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # 记录训练日志到TensorBoard
            if self.global_step % self.config.training.log_interval == 0:
                self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/recon_loss', outputs['recon_loss'].item(), self.global_step)
                self.writer.add_scalar('train/vq_loss', outputs['vq_loss'].item(), self.global_step)
                self.writer.add_scalar('train/codebook_utilization', codebook_util, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

                # 详细日志输出
                if self.global_step % (self.config.training.log_interval * 10) == 0:
                    print(f"\n📊 Step {self.global_step} 详细统计:")
                    print(f"   • 总损失: {loss.item():.6f}")
                    print(f"   • 重构损失: {outputs['recon_loss'].item():.6f}")
                    print(f"   • VQ损失: {outputs['vq_loss'].item():.6f}")
                    print(f"   • 码本利用率: {codebook_util:.3f}")
                    print(f"   • 学习率: {self.optimizer.param_groups[0]['lr']:.2e}")

            # 定期验证
            if self.global_step % self.config.training.val_interval == 0 and self.global_step > 0:
                print(f"\n🔍 执行验证 (Step {self.global_step})...")
                val_loss = self.validate()
                self.writer.add_scalar('val/total_loss', val_loss, self.global_step)

                print(f"   验证损失: {val_loss:.6f}")

                # 早停检查
                if val_loss < self.best_val_loss - self.config.training.early_stopping_min_delta:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                    print(f"   🎉 新的最佳验证损失! 改进: {improvement:.6f}")
                else:
                    self.patience_counter += 1
                    print(f"   ⏳ 验证损失未改进，耐心计数: {self.patience_counter}/{self.config.training.early_stopping_patience}")

            # 定期保存检查点
            if self.global_step % self.config.training.save_interval == 0 and self.global_step > 0:
                print(f"\n💾 保存检查点 (Step {self.global_step})...")
                self.save_checkpoint()

            self.global_step += 1

        # 计算epoch平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_vq_loss = total_vq_loss / len(self.train_loader)

        epoch_time = time.time() - epoch_start_time

        print(f"\n📈 Epoch {self.epoch + 1} 完成:")
        print(f"   • 平均总损失: {avg_loss:.6f}")
        print(f"   • 平均重构损失: {avg_recon_loss:.6f}")
        print(f"   • 平均VQ损失: {avg_vq_loss:.6f}")
        print(f"   • 训练时间: {epoch_time:.1f}秒")
        print(f"   • 码本利用率: {self.model.get_codebook_utilization():.3f}")

        return avg_loss, avg_recon_loss, avg_vq_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                total_loss += outputs['total_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.training.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.training.output_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """完整训练流程"""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_loss, recon_loss, vq_loss = self.train_epoch()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 验证
            val_loss = self.validate()
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"recon_loss={recon_loss:.4f}, vq_loss={vq_loss:.4f}")
            
            # 早停检查
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # 保存最终模型
        final_path = os.path.join(self.config.training.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Semantic VQ-VAE")
    parser.add_argument("--config", choices=["default", "small", "large"], default="default")
    parser.add_argument("--data_dir", default="./data/processed")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # 获取配置
    if args.config == "small":
        config = get_small_config()
    elif args.config == "large":
        config = get_large_config()
    else:
        config = get_default_config()
    
    # 更新配置
    config.training.data_dir = args.data_dir
    config.training.output_dir = args.output_dir
    config.training.device = args.device
    config.training.resume_from_checkpoint = args.resume
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.training.device = "cpu"
    
    # 保存配置
    save_config(config, os.path.join(config.training.output_dir, "config.json"))
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
