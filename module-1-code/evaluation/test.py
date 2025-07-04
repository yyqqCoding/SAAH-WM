"""
VQ-VAE测试和评估脚本
"""

import os
import sys
import torch
import torch.nn.functional as F
import argparse
import json
from typing import List, Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae_model import SemanticVQVAE
from utils.clip_utils import CLIPTextEncoder, compute_reconstruction_metrics
from training.config import load_config


class VQVAETester:
    """VQ-VAE测试器"""
    
    def __init__(self, model_path, config_path=None, device="cuda"):
        """
        初始化VQ-VAE测试器

        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径
            device: 计算设备
        """
        print("=" * 60)
        print("🧪 初始化SAAH-WM模块一测试器")
        print("=" * 60)

        self.device = torch.device(device)
        print(f"🖥️ 使用设备: {self.device}")

        # 加载配置
        print("📋 加载配置...")
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
            print(f"   ✓ 从文件加载配置: {config_path}")
        else:
            # 使用默认配置
            from training.config import get_default_config
            self.config = get_default_config()
            print("   ✓ 使用默认配置")

        # 创建模型
        print("🏗️ 创建模型...")
        self.model = SemanticVQVAE(
            input_dim=self.config.model.input_dim,
            latent_dim=self.config.model.latent_dim,
            num_embeddings=self.config.model.num_embeddings,
            commitment_cost=self.config.model.commitment_cost,
            decay=self.config.model.decay,
            dropout=self.config.model.dropout
        ).to(self.device)

        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✓ 模型参数数量: {model_params:,}")

        # 加载权重
        print("📦 加载模型权重...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    print(f"   ✓ 加载检查点 (Epoch {checkpoint['epoch']})")
                if 'best_val_loss' in checkpoint:
                    print(f"   ✓ 最佳验证损失: {checkpoint['best_val_loss']:.6f}")
            else:
                self.model.load_state_dict(checkpoint)
            print(f"   ✓ 成功加载权重: {model_path}")
        except Exception as e:
            print(f"   ❌ 加载权重失败: {e}")
            raise

        self.model.eval()

        # 初始化CLIP编码器
        print("🔤 初始化CLIP编码器...")
        try:
            self.clip_encoder = CLIPTextEncoder(device=device)
            print("   ✓ CLIP编码器就绪")
        except Exception as e:
            print(f"   ❌ CLIP编码器初始化失败: {e}")
            raise

        # 计算压缩参数
        self.num_bits = (self.config.model.num_embeddings - 1).bit_length()
        compression_ratio = (self.config.model.input_dim * 32) / self.num_bits

        print("📊 压缩参数:")
        print(f"   • 码本大小: {self.config.model.num_embeddings}")
        print(f"   • 压缩比特数: {self.num_bits}")
        print(f"   • 压缩比: {compression_ratio:.1f}:1")
        print(f"   • 存储节省: {(1 - self.num_bits / (self.config.model.input_dim * 32)) * 100:.2f}%")
        print("=" * 60)
    
    def prompt_to_bits(self, prompt: str) -> str:
        """
        将文本提示转换为二进制串
        Args:
            prompt: 文本提示
        Returns:
            str: 二进制串
        """
        # 获取CLIP向量
        clip_vector = self.clip_encoder.encode_text(prompt)
        
        # 获取VQ-VAE索引
        with torch.no_grad():
            indices = self.model.encode(clip_vector)
        
        # 转换为二进制
        index = indices[0].item()
        binary_str = format(index, f'0{self.num_bits}b')
        
        return binary_str
    
    def bits_to_reconstructed_vector(self, bits: str) -> torch.Tensor:
        """
        将二进制串转换回重构的语义向量
        Args:
            bits: 二进制串
        Returns:
            torch.Tensor: 重构的语义向量
        """
        # 转换为索引
        index = int(bits, 2)
        indices = torch.tensor([index], device=self.device)
        
        # 重构向量
        with torch.no_grad():
            reconstructed = self.model.decode_from_indices(indices)
        
        return reconstructed
    
    def test_single_prompt(self, prompt: str) -> Dict:
        """
        测试单个提示的完整流水线
        Args:
            prompt: 文本提示
        Returns:
            dict: 测试结果
        """
        # 原始向量
        original_vector = self.clip_encoder.encode_text(prompt)
        
        # 压缩为比特串
        bits = self.prompt_to_bits(prompt)
        
        # 重构向量
        reconstructed_vector = self.bits_to_reconstructed_vector(bits)
        
        # 计算指标
        metrics = compute_reconstruction_metrics(original_vector, reconstructed_vector)
        
        return {
            'prompt': prompt,
            'bits': bits,
            'index': int(bits, 2),
            'metrics': metrics,
            'original_vector': original_vector.cpu(),
            'reconstructed_vector': reconstructed_vector.cpu()
        }
    
    def test_consistency(self, prompt: str, num_tests: int = 10) -> Dict:
        """
        测试重构的一致性
        Args:
            prompt: 文本提示
            num_tests: 测试次数
        Returns:
            dict: 一致性测试结果
        """
        bits = self.prompt_to_bits(prompt)
        reconstructed_vectors = []
        
        # 多次重构
        for _ in range(num_tests):
            reconstructed = self.bits_to_reconstructed_vector(bits)
            reconstructed_vectors.append(reconstructed.cpu())
        
        # 检查一致性
        first_vector = reconstructed_vectors[0]
        all_identical = all(
            torch.allclose(vec, first_vector, atol=1e-6) 
            for vec in reconstructed_vectors
        )
        
        # 计算最大差异
        max_diff = 0
        for vec in reconstructed_vectors[1:]:
            diff = torch.max(torch.abs(vec - first_vector)).item()
            max_diff = max(max_diff, diff)
        
        return {
            'prompt': prompt,
            'bits': bits,
            'all_identical': all_identical,
            'max_difference': max_diff,
            'num_tests': num_tests
        }
    
    def test_codebook_coverage(self, test_prompts: List[str]) -> Dict:
        """
        测试码本覆盖率
        Args:
            test_prompts: 测试提示列表
        Returns:
            dict: 码本覆盖率结果
        """
        used_indices = set()
        
        for prompt in test_prompts:
            bits = self.prompt_to_bits(prompt)
            index = int(bits, 2)
            used_indices.add(index)
        
        coverage = len(used_indices) / self.config.model.num_embeddings
        
        return {
            'total_embeddings': self.config.model.num_embeddings,
            'used_embeddings': len(used_indices),
            'coverage_ratio': coverage,
            'used_indices': sorted(list(used_indices))
        }
    
    def benchmark_performance(self, test_prompts: List[str]) -> Dict:
        """
        性能基准测试
        Args:
            test_prompts: 测试提示列表
        Returns:
            dict: 性能测试结果
        """
        import time
        
        # 编码性能测试
        start_time = time.time()
        for prompt in test_prompts:
            self.prompt_to_bits(prompt)
        encoding_time = time.time() - start_time
        
        # 解码性能测试
        bits_list = [self.prompt_to_bits(prompt) for prompt in test_prompts]
        start_time = time.time()
        for bits in bits_list:
            self.bits_to_reconstructed_vector(bits)
        decoding_time = time.time() - start_time
        
        return {
            'num_prompts': len(test_prompts),
            'encoding_time': encoding_time,
            'decoding_time': decoding_time,
            'avg_encoding_time': encoding_time / len(test_prompts),
            'avg_decoding_time': decoding_time / len(test_prompts)
        }
    
    def comprehensive_test(self, test_prompts: List[str]) -> Dict:
        """
        综合测试
        Args:
            test_prompts: 测试提示列表
        Returns:
            dict: 综合测试结果
        """
        results = {
            'model_info': {
                'num_embeddings': self.config.model.num_embeddings,
                'latent_dim': self.config.model.latent_dim,
                'num_bits': self.num_bits,
                'codebook_utilization': self.model.get_codebook_utilization()
            },
            'individual_tests': [],
            'consistency_tests': [],
            'codebook_coverage': None,
            'performance_benchmark': None,
            'summary_metrics': {}
        }
        
        # 个别测试
        print("Running individual tests...")
        for prompt in test_prompts:
            result = self.test_single_prompt(prompt)
            results['individual_tests'].append(result)
        
        # 一致性测试（选择前5个提示）
        print("Running consistency tests...")
        for prompt in test_prompts[:5]:
            result = self.test_consistency(prompt)
            results['consistency_tests'].append(result)
        
        # 码本覆盖率测试
        print("Testing codebook coverage...")
        results['codebook_coverage'] = self.test_codebook_coverage(test_prompts)
        
        # 性能基准测试
        print("Running performance benchmark...")
        results['performance_benchmark'] = self.benchmark_performance(test_prompts)
        
        # 汇总指标
        cos_sims = [test['metrics']['cosine_similarity'] for test in results['individual_tests']]
        mse_losses = [test['metrics']['mse_loss'] for test in results['individual_tests']]
        
        results['summary_metrics'] = {
            'avg_cosine_similarity': sum(cos_sims) / len(cos_sims),
            'min_cosine_similarity': min(cos_sims),
            'max_cosine_similarity': max(cos_sims),
            'avg_mse_loss': sum(mse_losses) / len(mse_losses),
            'all_consistent': all(test['all_identical'] for test in results['consistency_tests'])
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test Semantic VQ-VAE")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--config_path", help="Path to config file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--output_dir", default="./test_results", help="Output directory")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建测试器
    tester = VQVAETester(args.model_path, args.config_path, args.device)
    
    # 测试提示
    test_prompts = [
        "a cat sitting on a chair",
        "a beautiful sunset over the ocean",
        "a person riding a bicycle in the park",
        "a red car driving on the highway",
        "a dog playing with a ball",
        "a mountain landscape with snow",
        "a city skyline at night",
        "a flower garden in spring",
        "a bird flying in the sky",
        "a book on a wooden table"
    ]
    
    # 运行综合测试
    print("Starting comprehensive test...")
    results = tester.comprehensive_test(test_prompts)
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, 'w') as f:
        # 转换tensor为list以便JSON序列化
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_tensors(results), f, indent=2)
    
    print(f"Test results saved to {results_path}")
    
    # 打印汇总
    print("\n=== Test Summary ===")
    print(f"Average cosine similarity: {results['summary_metrics']['avg_cosine_similarity']:.4f}")
    print(f"Minimum cosine similarity: {results['summary_metrics']['min_cosine_similarity']:.4f}")
    print(f"Average MSE loss: {results['summary_metrics']['avg_mse_loss']:.6f}")
    print(f"All reconstructions consistent: {results['summary_metrics']['all_consistent']}")
    print(f"Codebook coverage: {results['codebook_coverage']['coverage_ratio']:.3f}")
    print(f"Codebook utilization: {results['model_info']['codebook_utilization']:.3f}")


if __name__ == "__main__":
    main()
