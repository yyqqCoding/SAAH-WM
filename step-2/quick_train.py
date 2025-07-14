"""
SAAH-WM Baseline 第二步 - 快速训练验证脚本

用于验证完整训练-测试流程的快速训练脚本。
生成测试图像，运行短时间训练，验证模型权重生成和推理流程。
"""

import os
import sys
import tempfile
import shutil
from PIL import Image
import numpy as np
import torch

def create_test_images(num_images=20, image_dir="test_images"):
    """创建测试图像"""
    print(f"🖼️  创建 {num_images} 张测试图像...")
    
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)
    
    for i in range(num_images):
        # 创建随机彩色图像
        np.random.seed(i)  # 确保可重复
        
        # 生成不同类型的测试图像
        if i % 4 == 0:
            # 渐变图像
            img_array = np.zeros((256, 256, 3), dtype=np.uint8)
            for x in range(256):
                for y in range(256):
                    img_array[y, x] = [x % 256, y % 256, (x + y) % 256]
        elif i % 4 == 1:
            # 随机噪声图像
            img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        elif i % 4 == 2:
            # 几何图案
            img_array = np.zeros((256, 256, 3), dtype=np.uint8)
            center = 128
            for x in range(256):
                for y in range(256):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    img_array[y, x] = [
                        int(128 + 127 * np.sin(dist / 10)),
                        int(128 + 127 * np.cos(dist / 15)),
                        int(128 + 127 * np.sin(dist / 20))
                    ]
        else:
            # 纯色块
            color = [(i * 50) % 256, (i * 80) % 256, (i * 120) % 256]
            img_array = np.full((256, 256, 3), color, dtype=np.uint8)
        
        # 保存图像
        img = Image.fromarray(img_array)
        img.save(os.path.join(image_dir, f"test_{i:03d}.jpg"))
    
    print(f"✅ 测试图像创建完成: {image_dir}")

def run_quick_training():
    """运行快速训练"""
    print("🚀 开始快速训练...")
    
    # 创建测试图像
    create_test_images(20)
    
    # 导入主程序
    from main import main
    import sys
    
    # 设置命令行参数
    original_argv = sys.argv
    sys.argv = [
        'quick_train.py',
        '--config', 'configs/quick_test_config.yaml',
        '--debug'
    ]
    
    try:
        # 运行训练
        main()
        print("✅ 快速训练完成")
        return True
    except Exception as e:
        print(f"❌ 快速训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 恢复命令行参数
        sys.argv = original_argv

def test_trained_models():
    """测试训练好的模型"""
    print("🧪 测试训练好的模型...")
    
    # 检查模型文件是否存在
    models_dir = "trained_models"
    expected_files = [
        "fragile_encoder.pth",
        "robust_encoder.pth", 
        "robust_decoder.pth",
        "fragile_decoder.pth"
    ]
    
    if not os.path.exists(models_dir):
        print(f"❌ 模型目录不存在: {models_dir}")
        return False
    
    missing_files = []
    for filename in expected_files:
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
        else:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ {filename}: {size_mb:.1f} MB")
    
    if missing_files:
        print(f"❌ 缺少模型文件: {missing_files}")
        return False
    
    # 测试模型加载和推理
    try:
        from models import FragileEncoder, RobustEncoder, RobustDecoder, FragileDecoder
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建模型（使用训练时的配置）
        fragile_encoder = FragileEncoder(latent_channels=4, hidden_dim=32, num_layers=2).to(device)
        robust_encoder = RobustEncoder(latent_channels=4, message_dim=32, hidden_dim=32, num_layers=2).to(device)
        robust_decoder = RobustDecoder(latent_channels=4, message_dim=32, hidden_dim=32, num_layers=2).to(device)
        fragile_decoder = FragileDecoder(latent_channels=4, hidden_dim=32, num_layers=2).to(device)
        
        # 加载权重
        fragile_encoder.load_state_dict(torch.load(os.path.join(models_dir, "fragile_encoder.pth"), map_location=device))
        robust_encoder.load_state_dict(torch.load(os.path.join(models_dir, "robust_encoder.pth"), map_location=device))
        robust_decoder.load_state_dict(torch.load(os.path.join(models_dir, "robust_decoder.pth"), map_location=device))
        fragile_decoder.load_state_dict(torch.load(os.path.join(models_dir, "fragile_decoder.pth"), map_location=device))
        
        print("✅ 所有模型权重加载成功")
        
        # 测试推理
        fragile_encoder.eval()
        robust_encoder.eval()
        robust_decoder.eval()
        fragile_decoder.eval()
        
        with torch.no_grad():
            # 创建测试数据
            batch_size = 1
            height, width = 16, 16
            message_dim = 32
            
            image_latent = torch.randn(batch_size, 4, height, width).to(device)
            base_watermark = torch.randn(batch_size, 4, height, width).to(device)
            message_bits = torch.randint(0, 2, (batch_size, message_dim)).float().to(device)
            
            # 完整推理流程
            print("🔄 执行完整推理流程...")
            
            # 步骤1: 脆弱水印嵌入
            watermarked_fragile = fragile_encoder(image_latent, base_watermark)
            print(f"  脆弱嵌入: {image_latent.shape} -> {watermarked_fragile.shape}")
            
            # 步骤2: 鲁棒信息嵌入
            watermarked_robust = robust_encoder(watermarked_fragile, message_bits)
            print(f"  鲁棒嵌入: {watermarked_fragile.shape} -> {watermarked_robust.shape}")
            
            # 步骤3: 鲁棒信息解码
            decoded_message = robust_decoder(watermarked_robust)
            print(f"  鲁棒解码: {watermarked_robust.shape} -> {decoded_message.shape}")
            
            # 步骤4: 脆弱水印解码
            recovered_image, recovered_watermark = fragile_decoder(watermarked_robust)
            print(f"  脆弱解码: {watermarked_robust.shape} -> {recovered_image.shape}, {recovered_watermark.shape}")
            
            # 计算简单指标
            from utils.metrics import calculate_psnr, calculate_bit_accuracy
            
            psnr = calculate_psnr(watermarked_robust, image_latent)
            bit_accuracy = calculate_bit_accuracy(decoded_message, message_bits)
            
            print(f"📊 推理结果:")
            print(f"  图像PSNR: {psnr:.2f} dB")
            print(f"  比特准确率: {bit_accuracy:.4f}")
            
        print("✅ 模型推理测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    
    # 删除测试图像
    if os.path.exists("test_images"):
        shutil.rmtree("test_images")
        print("✅ 测试图像已删除")

def main():
    """主函数"""
    print("🎯 SAAH-WM Baseline 第二步 - 快速训练验证")
    print("=" * 60)
    
    success = True
    
    try:
        # 步骤1: 运行快速训练
        if not run_quick_training():
            success = False
        
        # 步骤2: 测试训练好的模型
        if success and not test_trained_models():
            success = False
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        success = False
    except Exception as e:
        print(f"\n❌ 发生未预期错误: {e}")
        success = False
    finally:
        # 清理
        cleanup()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 快速训练验证成功！")
        print("✅ 训练-测试完整流程验证通过")
        print("✅ 模型权重生成和加载正常")
        print("✅ 推理流程运行正常")
    else:
        print("❌ 快速训练验证失败")
        print("请检查错误信息并修复问题")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
