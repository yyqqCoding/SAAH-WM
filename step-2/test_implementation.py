"""
SAAH-WM Baseline 第二步 - 实现测试脚本

本脚本用于测试所有已实现的模块，确保代码正确性。

测试内容：
1. 模型创建和前向传播
2. 损失函数计算
3. 攻击层功能
4. 评估指标计算
5. 数据加载器
6. 训练器基本功能
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import tempfile
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_models():
    """测试四个核心模型"""
    print("🧪 测试核心模型...")
    
    try:
        from models import FragileEncoder, RobustEncoder, RobustDecoder, FragileDecoder
        
        # 测试参数
        batch_size = 2
        latent_channels = 4
        height, width = 64, 64
        message_dim = 64
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        
        # 创建模型
        fragile_encoder = FragileEncoder(latent_channels=latent_channels).to(device)
        robust_encoder = RobustEncoder(latent_channels=latent_channels, message_dim=message_dim).to(device)
        robust_decoder = RobustDecoder(latent_channels=latent_channels, message_dim=message_dim).to(device)
        fragile_decoder = FragileDecoder(latent_channels=latent_channels).to(device)
        
        print("✅ 模型创建成功")
        
        # 创建测试数据
        image_latent = torch.randn(batch_size, latent_channels, height, width).to(device)
        base_watermark = torch.randn(batch_size, latent_channels, height, width).to(device)
        message_bits = torch.randn(batch_size, message_dim).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            # 脆弱编码
            watermarked_fragile = fragile_encoder(image_latent, base_watermark)
            print(f"✅ 脆弱编码器输出: {watermarked_fragile.shape}")
            
            # 鲁棒编码
            watermarked_robust = robust_encoder(watermarked_fragile, message_bits)
            print(f"✅ 鲁棒编码器输出: {watermarked_robust.shape}")
            
            # 鲁棒解码
            decoded_message = robust_decoder(watermarked_robust)
            print(f"✅ 鲁棒解码器输出: {decoded_message.shape}")
            
            # 脆弱解码
            recovered_image, recovered_watermark = fragile_decoder(watermarked_robust)
            print(f"✅ 脆弱解码器输出: {recovered_image.shape}, {recovered_watermark.shape}")
        
        print("✅ 所有模型前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n🧪 测试损失函数...")
    
    try:
        from utils.loss_functions import WatermarkLoss
        
        # 创建损失函数
        criterion = WatermarkLoss()
        
        # 创建测试数据
        batch_size = 2
        height, width = 64, 64
        message_dim = 64
        
        outputs = {
            'watermarked_image': torch.randn(batch_size, 4, height, width),
            'decoded_message': torch.randn(batch_size, message_dim),
            'recovered_image': torch.randn(batch_size, 4, height, width),
            'recovered_watermark': torch.randn(batch_size, 4, height, width)
        }
        
        targets = {
            'original_image': torch.randn(batch_size, 4, height, width),
            'target_message': torch.randint(0, 2, (batch_size, message_dim)).float(),
            'target_watermark': torch.randn(batch_size, 4, height, width)
        }
        
        # 计算损失
        total_loss, loss_details = criterion(outputs, targets)
        
        print(f"✅ 总损失: {total_loss.item():.6f}")
        print(f"✅ 损失详情: {loss_details}")
        
        return True
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False


def test_attack_layers():
    """测试攻击层"""
    print("\n🧪 测试攻击层...")
    
    try:
        from utils.attack_layers import AttackLayer
        
        # 攻击配置
        attack_config = {
            'jpeg_compression': {
                'enabled': True,
                'quality_range': [70, 90],
                'probability': 1.0
            },
            'gaussian_noise': {
                'enabled': True,
                'sigma_range': [0.01, 0.05],
                'probability': 1.0
            }
        }
        
        # 创建攻击层
        attack_layer = AttackLayer(attack_config)
        
        # 测试数据
        test_image = torch.randn(2, 4, 64, 64)
        
        # 应用攻击
        attacked_image = attack_layer(test_image)
        
        # 计算差异
        diff = torch.abs(attacked_image - test_image).mean()
        
        print(f"✅ 攻击层输出形状: {attacked_image.shape}")
        print(f"✅ 平均差异: {diff.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 攻击层测试失败: {e}")
        return False


def test_metrics():
    """测试评估指标"""
    print("\n🧪 测试评估指标...")
    
    try:
        from utils.metrics import WatermarkMetrics, calculate_psnr, calculate_ssim
        
        # 创建测试数据
        img1 = torch.randn(2, 4, 64, 64)
        img2 = img1 + torch.randn_like(img1) * 0.1  # 添加少量噪声
        
        # 测试PSNR
        psnr = calculate_psnr(img1, img2)
        print(f"✅ PSNR: {psnr:.2f} dB")
        
        # 测试SSIM
        ssim = calculate_ssim(img1, img2)
        print(f"✅ SSIM: {ssim:.4f}")
        
        # 测试指标计算器
        metrics_calculator = WatermarkMetrics()
        
        outputs = {
            'watermarked_image': img2,
            'decoded_message': torch.sigmoid(torch.randn(2, 64)),
            'recovered_image': img1,
            'recovered_watermark': torch.randn(2, 4, 64, 64)
        }
        
        targets = {
            'original_image': img1,
            'target_message': torch.randint(0, 2, (2, 64)).float(),
            'target_watermark': torch.randn(2, 4, 64, 64)
        }
        
        current_metrics = metrics_calculator.update(outputs, targets)
        print(f"✅ 当前指标: {current_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估指标测试失败: {e}")
        return False


def test_data_loader():
    """测试数据加载器"""
    print("\n🧪 测试数据加载器...")
    
    try:
        from data.coco_dataloader import COCOWatermarkDataset, create_dataloader
        
        # 创建临时测试目录和图像
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试图像
            for i in range(5):
                test_image = Image.new('RGB', (256, 256), color=(i*50, 100, 150))
                test_image.save(os.path.join(temp_dir, f'test_{i}.jpg'))
            
            # 创建数据集
            dataset = COCOWatermarkDataset(
                coco_path=temp_dir,
                image_size=256,
                crop_size=200,
                use_step1_modules=False,  # 使用模拟数据
                latent_size=32
            )
            
            print(f"✅ 数据集大小: {len(dataset)}")
            
            # 测试单个样本
            sample = dataset[0]
            print(f"✅ 样本键: {list(sample.keys())}")
            
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # 创建数据加载器
            dataloader = create_dataloader(
                coco_path=temp_dir,
                batch_size=2,
                num_workers=0,
                use_step1_modules=False
            )
            
            # 测试批次加载
            batch = next(iter(dataloader))
            print(f"✅ 批次加载成功，批次大小: {batch['image_latent'].shape[0]}")
            
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False


def test_trainer_basic():
    """测试训练器基本功能"""
    print("\n🧪 测试训练器基本功能...")
    
    try:
        # 创建简化配置
        config = {
            'model': {
                'latent_channels': 4,
                'fragile_encoder': {'hidden_dim': 64, 'num_layers': 3},
                'robust_encoder': {'message_dim': 32, 'hidden_dim': 64, 'num_layers': 3},
                'robust_decoder': {'message_dim': 32, 'hidden_dim': 64, 'num_layers': 3},
                'fragile_decoder': {'hidden_dim': 64, 'num_layers': 3}
            },
            'training': {
                'loss_weights': {'image_loss': 1.0, 'robust_loss': 10.0, 'fragile_loss': 5.0},
                'optimizer': 'adamw',
                'learning_rate': 1e-4,
                'weight_decay': 1e-6,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'lr_scheduler': 'cosine',
                'num_epochs': 2,
                'min_lr': 1e-6,
                'gradient_clip_norm': 1.0
            },
            'attacks': {
                'jpeg_compression': {'enabled': True, 'quality_range': [70, 90], 'probability': 0.5},
                'gaussian_noise': {'enabled': True, 'sigma_range': [0.01, 0.05], 'probability': 0.5}
            },
            'checkpoints': {
                'save_dir': 'test_checkpoints',
                'performance_thresholds': {'psnr': 30.0, 'ssim': 0.8, 'bit_accuracy': 0.8}
            },
            'logging': {'log_dir': 'test_logs', 'level': 'INFO'}
        }
        
        from core.trainer import WatermarkTrainer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = WatermarkTrainer(config, device)
        
        print("✅ 训练器创建成功")
        
        # 创建模拟批次数据
        batch = {
            'image_latent': torch.randn(2, 4, 32, 32),
            'base_watermark': torch.randn(2, 4, 32, 32),
            'message_bits': torch.randint(0, 2, (2, 32)).float()
        }
        
        # 测试前向传播
        outputs = trainer.forward_pass(batch)
        print(f"✅ 前向传播成功，输出键: {list(outputs.keys())}")
        
        # 测试训练步骤
        step_metrics = trainer.train_step(batch)
        print(f"✅ 训练步骤成功，指标: {step_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试SAAH-WM Baseline第二步实现")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_functions = [
        ("核心模型", test_models),
        ("损失函数", test_loss_functions),
        ("攻击层", test_attack_layers),
        ("评估指标", test_metrics),
        ("数据加载器", test_data_loader),
        ("训练器基本功能", test_trainer_basic)
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            test_results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！代码实现正确。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
