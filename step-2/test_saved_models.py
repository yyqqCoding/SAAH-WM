"""
测试保存的模型权重
"""

import torch
import os
import glob
from models import FragileEncoder, RobustEncoder, RobustDecoder, FragileDecoder

def test_saved_models():
    """测试保存的模型"""
    print("🧪 测试保存的模型权重...")
    
    # 查找最新的检查点文件
    checkpoint_files = glob.glob("checkpoints/checkpoint_*.pth")
    if not checkpoint_files:
        print("❌ 没有找到检查点文件")
        return False
    
    # 使用最新的检查点
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"📁 加载检查点: {latest_checkpoint}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # 加载检查点
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        print(f"✅ 检查点加载成功")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Global Step: {checkpoint['global_step']}")
        
        # 创建模型（使用训练时的配置）
        fragile_encoder = FragileEncoder(latent_channels=4, hidden_dim=32, num_layers=2).to(device)
        robust_encoder = RobustEncoder(latent_channels=4, message_dim=32, hidden_dim=32, num_layers=2).to(device)
        robust_decoder = RobustDecoder(latent_channels=4, message_dim=32, hidden_dim=32, num_layers=2).to(device)
        fragile_decoder = FragileDecoder(latent_channels=4, hidden_dim=32, num_layers=2).to(device)
        
        # 加载权重
        fragile_encoder.load_state_dict(checkpoint['fragile_encoder_state_dict'])
        robust_encoder.load_state_dict(checkpoint['robust_encoder_state_dict'])
        robust_decoder.load_state_dict(checkpoint['robust_decoder_state_dict'])
        fragile_decoder.load_state_dict(checkpoint['fragile_decoder_state_dict'])
        
        print("✅ 所有模型权重加载成功")
        
        # 设置为评估模式
        fragile_encoder.eval()
        robust_encoder.eval()
        robust_decoder.eval()
        fragile_decoder.eval()
        
        # 测试推理
        with torch.no_grad():
            # 创建测试数据
            batch_size = 1
            height, width = 16, 16
            message_dim = 32
            
            image_latent = torch.randn(batch_size, 4, height, width).to(device)
            base_watermark = torch.randn(batch_size, 4, height, width).to(device)
            message_bits = torch.randint(0, 2, (batch_size, message_dim)).float().to(device)
            
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
            
            # 保存单独的模型权重文件
            models_dir = "trained_models"
            os.makedirs(models_dir, exist_ok=True)
            
            torch.save(fragile_encoder.state_dict(), os.path.join(models_dir, "fragile_encoder.pth"))
            torch.save(robust_encoder.state_dict(), os.path.join(models_dir, "robust_encoder.pth"))
            torch.save(robust_decoder.state_dict(), os.path.join(models_dir, "robust_decoder.pth"))
            torch.save(fragile_decoder.state_dict(), os.path.join(models_dir, "fragile_decoder.pth"))
            
            print(f"💾 模型权重已保存到: {models_dir}")
            for filename in ["fragile_encoder.pth", "robust_encoder.pth", "robust_decoder.pth", "fragile_decoder.pth"]:
                filepath = os.path.join(models_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {filename}: {size_mb:.1f} MB")
        
        print("✅ 模型测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_saved_models()
    if success:
        print("\n🎉 模型权重验证成功！")
        print("✅ 训练流程完整可用")
        print("✅ 模型权重保存正常")
        print("✅ 推理流程运行正常")
        print("✅ 四个.pth文件已生成")
    else:
        print("\n❌ 模型权重验证失败")
