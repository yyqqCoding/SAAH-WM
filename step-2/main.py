"""
SAAH-WM Baseline 第二步 - 主程序入口

本程序实现完整的水印编码器/解码器训练流程。

使用方法：
    python main.py --config configs/train_config.yaml
    python main.py --config configs/train_config.yaml --resume checkpoints/checkpoint.pth
    python main.py --config configs/train_config.yaml --test_only

功能特点：
- 支持从配置文件加载训练参数
- 支持断点续训
- 支持仅测试模式
- 完整的日志记录和性能监控
- 自动保存最佳模型权重
"""

import argparse
import os
import sys
import yaml
import torch
import logging
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from core.trainer import WatermarkTrainer
from data.coco_dataloader import create_dataloader
from utils.logger_config import setup_training_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config


def setup_device(config: Dict[str, Any]) -> str:
    """
    设置训练设备
    
    Args:
        config (Dict[str, Any]): 配置字典
        
    Returns:
        str: 设备名称
    """
    if config.get('device', 'cuda') == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU训练: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("使用CPU训练")
        
    return device


def create_dataloaders(config: Dict[str, Any]) -> tuple:
    """
    创建数据加载器
    
    Args:
        config (Dict[str, Any]): 配置字典
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    dataset_config = config['dataset']
    
    # 训练数据加载器
    train_dataloader = create_dataloader(
        coco_path=dataset_config['coco_path'],
        batch_size=dataset_config['batch_size'],
        num_workers=dataset_config['num_workers'],
        pin_memory=dataset_config['pin_memory'],
        shuffle=True,
        split='train',
        image_size=dataset_config['image_size'],
        crop_size=dataset_config['crop_size'],
        use_step1_modules=True  # 使用第一步模块
    )
    
    # 验证数据加载器（使用部分训练数据）
    val_dataloader = create_dataloader(
        coco_path=dataset_config['coco_path'],
        batch_size=dataset_config['batch_size'],
        num_workers=dataset_config['num_workers'],
        pin_memory=dataset_config['pin_memory'],
        shuffle=False,
        split='val',
        image_size=dataset_config['image_size'],
        crop_size=dataset_config['crop_size'],
        use_step1_modules=True
    )
    
    return train_dataloader, val_dataloader


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SAAH-WM Baseline 第二步训练程序')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--test_only', action='store_true',
                       help='仅进行测试，不训练')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print("🔧 加载配置文件...")
        config = load_config(args.config)
        
        # 设置调试模式
        if args.debug or config.get('debug', {}).get('enabled', False):
            config['logging']['level'] = 'DEBUG'
            print("🐛 调试模式已启用")
        
        # 设置日志
        training_logger = setup_training_logger(
            log_dir=config['logging']['log_dir'],
            log_level=config['logging']['level']
        )
        
        print("📝 日志系统初始化完成")
        
        # 设置设备
        device = setup_device(config)
        
        # 设置随机种子
        if 'seed' in config:
            torch.manual_seed(config['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config['seed'])
            print(f"🎲 随机种子设置为: {config['seed']}")
        
        # 创建数据加载器
        print("📊 创建数据加载器...")
        train_dataloader, val_dataloader = create_dataloaders(config)
        
        print(f"训练集大小: {len(train_dataloader.dataset)}")
        print(f"验证集大小: {len(val_dataloader.dataset)}")
        print(f"批次大小: {config['dataset']['batch_size']}")
        print(f"训练批次数: {len(train_dataloader)}")
        print(f"验证批次数: {len(val_dataloader)}")
        
        # 创建训练器
        print("🤖 初始化训练器...")
        trainer = WatermarkTrainer(config, device)
        
        # 打印模型摘要
        print("\n" + trainer.get_model_summary())
        
        # 恢复训练（如果指定）
        if args.resume:
            print(f"🔄 从检查点恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        if args.test_only:
            # 仅测试模式
            print("🧪 开始测试...")
            val_metrics = trainer.validate_epoch(val_dataloader, training_logger)
            
            print("\n测试结果:")
            for key, value in val_metrics.items():
                if 'accuracy' in key or 'rate' in key:
                    print(f"  {key}: {value:.4f}")
                elif 'loss' in key:
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value:.2f}")
                    
        else:
            # 训练模式
            print("🚀 开始训练...")
            
            # 检查快速运行模式
            if config.get('debug', {}).get('fast_run', False):
                print("⚡ 快速运行模式已启用")
                # 限制数据集大小
                fast_samples = config['debug']['fast_run_samples']
                train_dataloader.dataset.image_files = train_dataloader.dataset.image_files[:fast_samples]
                val_dataloader.dataset.image_files = val_dataloader.dataset.image_files[:fast_samples//10]
                print(f"数据集已限制为 {fast_samples} 个训练样本")
            
            # 开始训练
            best_metrics = trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                training_logger=training_logger
            )
            
            print("\n🎉 训练完成!")
            print("最佳性能:")
            for key, value in best_metrics.items():
                if key != 'epoch':
                    if 'accuracy' in key or 'rate' in key:
                        print(f"  {key}: {value:.4f}")
                    elif 'loss' in key:
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value:.2f}")
            
            # 检查性能阈值
            thresholds = config['checkpoints']['performance_thresholds']
            print("\n🎯 性能阈值检查:")
            
            for metric, threshold in thresholds.items():
                if metric in best_metrics:
                    value = best_metrics[metric]
                    status = "✅ 达标" if value >= threshold else "❌ 未达标"
                    print(f"  {metric}: {value:.4f} (阈值: {threshold}) {status}")
            
            # 输出模型文件位置
            models_dir = os.path.join(config['checkpoints']['save_dir'], 'best_models')
            if os.path.exists(models_dir):
                print(f"\n💾 最佳模型权重已保存到: {models_dir}")
                print("包含以下文件:")
                for filename in ['fragile_encoder.pth', 'robust_encoder.pth', 
                               'robust_decoder.pth', 'fragile_decoder.pth']:
                    filepath = os.path.join(models_dir, filename)
                    if os.path.exists(filepath):
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        print(f"  - {filename} ({size_mb:.1f} MB)")
        
    except KeyboardInterrupt:
        print("\n⚠️  程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        logging.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
