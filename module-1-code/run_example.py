"""
快速运行示例脚本
演示如何使用语义VQ-VAE模型
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vqvae_model import create_semantic_vqvae
from utils.clip_utils import CLIPTextEncoder, test_compression_pipeline


def demo_model_creation():
    """演示模型创建"""
    print("=== 模型创建演示 ===")
    
    # 创建模型
    model = create_semantic_vqvae()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入维度: {model.input_dim}")
    print(f"潜在维度: {model.latent_dim}")
    print(f"码本大小: {model.num_embeddings}")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, model.input_dim)
    
    with torch.no_grad():
        outputs = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"重构形状: {outputs['x_recon'].shape}")
    print(f"总损失: {outputs['total_loss'].item():.4f}")
    print(f"重构损失: {outputs['recon_loss'].item():.4f}")
    print(f"VQ损失: {outputs['vq_loss'].item():.4f}")
    print()


def demo_clip_encoding():
    """演示CLIP编码"""
    print("=== CLIP编码演示 ===")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 创建CLIP编码器
        clip_encoder = CLIPTextEncoder(device=device)
        
        # 测试文本
        test_prompts = [
            "a cat sitting on a chair",
            "a beautiful sunset over the ocean",
            "a person riding a bicycle"
        ]
        
        # 编码文本
        for prompt in test_prompts:
            vector = clip_encoder.encode_text(prompt)
            print(f"文本: '{prompt}'")
            print(f"向量形状: {vector.shape}")
            print(f"向量范数: {torch.norm(vector).item():.4f}")
            print()
            
    except Exception as e:
        print(f"CLIP编码失败: {e}")
        print("请确保已安装transformers库并有网络连接")
        print()


def demo_compression_pipeline():
    """演示压缩流水线"""
    print("=== 压缩流水线演示 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 创建模型和编码器
        model = create_semantic_vqvae()
        model.to(device)
        model.eval()
        
        clip_encoder = CLIPTextEncoder(device=device)
        
        # 测试提示
        test_prompts = [
            "a red car on the street",
            "a dog playing in the park"
        ]
        
        # 运行测试
        results = test_compression_pipeline(test_prompts, model, clip_encoder)
        
        for result in results:
            print(f"文本: '{result['prompt']}'")
            print(f"压缩比特串: {result['bits']}")
            print(f"余弦相似度: {result['metrics']['cosine_similarity']:.4f}")
            print(f"MSE损失: {result['metrics']['mse_loss']:.6f}")
            print()
            
    except Exception as e:
        print(f"压缩流水线演示失败: {e}")
        print("这是正常的，因为模型还未训练")
        print()


def demo_data_flow():
    """演示数据流"""
    print("=== 数据流演示 ===")
    
    # 模拟CLIP向量
    batch_size = 8
    clip_dim = 768
    clip_vectors = torch.randn(batch_size, clip_dim)
    
    # 创建模型
    model = create_semantic_vqvae()
    model.eval()
    
    print(f"输入CLIP向量: {clip_vectors.shape}")
    
    with torch.no_grad():
        # 编码
        z_e = model.encoder(clip_vectors)
        print(f"编码器输出: {z_e.shape}")
        
        # 量化
        z_q, vq_loss, indices = model.quantizer(z_e)
        print(f"量化后向量: {z_q.shape}")
        print(f"量化索引: {indices.shape}")
        print(f"索引范围: {indices.min().item()} - {indices.max().item()}")
        
        # 解码
        x_recon = model.decoder(z_q)
        print(f"重构向量: {x_recon.shape}")
        
        # 计算压缩比
        original_bits = clip_dim * 32  # 假设32位浮点数
        compressed_bits = (model.num_embeddings - 1).bit_length()
        compression_ratio = original_bits / compressed_bits
        
        print(f"原始存储: {original_bits} bits")
        print(f"压缩存储: {compressed_bits} bits")
        print(f"压缩比: {compression_ratio:.1f}:1")
        print()


def main():
    """主函数"""
    print("SAAH-WM 模块一：语义VQ-VAE 演示")
    print("=" * 50)
    print()
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    print()
    
    # 运行演示
    demo_model_creation()
    demo_data_flow()
    demo_clip_encoding()
    demo_compression_pipeline()
    
    print("演示完成！")
    print()
    print("下一步:")
    print("1. 运行数据预处理: python data/preprocess_data.py")
    print("2. 开始训练: python training/train.py --config small")
    print("3. 测试模型: python evaluation/test.py --model_path <model_path>")


if __name__ == "__main__":
    main()
