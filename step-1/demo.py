"""
SAAH-WM Baseline 第一步演示程序

修复相对导入问题的完整演示版本。

作者：SAAH-WM团队
"""

import os
import sys
import torch
import time

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.model_loader import ModelLoader
from utils.logger_config import get_default_logger
from utils.common_utils import ensure_dir, set_random_seed


def test_semantic_hash_with_models():
    """测试语义哈希生成（需要CLIP模型）"""
    logger = get_default_logger()
    logger.info("=" * 40)
    logger.info("测试语义哈希生成")
    logger.info("=" * 40)
    
    try:
        # 加载模型
        model_loader = ModelLoader()
        clip_model, clip_processor, _ = model_loader.load_all_models()
        
        # 导入语义哈希生成器
        from core.semantic_hash import SemanticHashGenerator
        
        # 创建生成器
        hash_generator = SemanticHashGenerator(clip_model, clip_processor)
        
        # 测试prompt
        test_prompt = "一只戴着宇航头盔的柴犬"
        logger.info(f"测试prompt: '{test_prompt}'")
        
        # 生成语义哈希
        start_time = time.time()
        c_bits = hash_generator.generate_semantic_hash(test_prompt)
        end_time = time.time()
        
        logger.info(f"语义哈希生成完成，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"生成的哈希: {c_bits[:32]}...{c_bits[-32:]}")
        
        # 验证一致性
        logger.info("验证哈希一致性...")
        is_consistent = hash_generator.verify_hash_consistency(test_prompt, num_tests=3)
        logger.info(f"一致性验证: {'通过' if is_consistent else '失败'}")
        
        # 清理模型
        model_loader.clear_models()
        
        return c_bits
        
    except Exception as e:
        logger.error(f"语义哈希测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_base_watermark(c_bits):
    """测试基准水印生成"""
    logger = get_default_logger()
    logger.info("=" * 40)
    logger.info("测试基准水印生成")
    logger.info("=" * 40)
    
    try:
        from core.base_watermark import BaseWatermarkGenerator
        
        # 创建生成器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        watermark_gen = BaseWatermarkGenerator(device)
        
        # 生成基准水印
        start_time = time.time()
        w_base = watermark_gen.generate_base_watermark(c_bits)
        end_time = time.time()
        
        logger.info(f"基准水印生成完成，耗时: {end_time - start_time:.4f}秒")
        logger.info(f"水印形状: {w_base.shape}")
        
        # 验证确定性
        logger.info("验证确定性生成...")
        is_deterministic = watermark_gen.verify_deterministic_generation(c_bits, num_tests=3)
        logger.info(f"确定性验证: {'通过' if is_deterministic else '失败'}")
        
        # 获取统计信息
        stats = watermark_gen.get_watermark_statistics(w_base)
        logger.info(f"水印统计: 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}")
        
        return w_base
        
    except Exception as e:
        logger.error(f"基准水印测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_message_packet(c_bits):
    """测试信息包生成"""
    logger = get_default_logger()
    logger.info("=" * 40)
    logger.info("测试信息包生成")
    logger.info("=" * 40)
    
    try:
        from core.message_packet import MessagePacketGenerator
        
        # 创建生成器
        packet_gen = MessagePacketGenerator()
        
        # 测试版权信息
        copyright_info = "UserID:12345"
        logger.info(f"版权信息: '{copyright_info}'")
        
        # 生成信息包
        start_time = time.time()
        packet = packet_gen.create_message_packet(c_bits, copyright_info)
        end_time = time.time()
        
        logger.info(f"信息包生成完成，耗时: {end_time - start_time:.4f}秒")
        logger.info(f"信息包长度: {len(packet)}位")
        
        # 验证完整性
        logger.info("验证信息包完整性...")
        is_valid = packet_gen.verify_packet_integrity(c_bits, copyright_info)
        logger.info(f"完整性验证: {'通过' if is_valid else '失败'}")
        
        # 获取配置信息
        packet_info = packet_gen.get_packet_info()
        logger.info(f"BCH参数: t={packet_info['bch_t']}, n={packet_info['bch_n']}")
        
        return packet
        
    except Exception as e:
        logger.error(f"信息包测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_attention_extraction_simple():
    """简化的注意力提取测试（不运行完整SD）"""
    logger = get_default_logger()
    logger.info("=" * 40)
    logger.info("测试注意力提取器（简化版）")
    logger.info("=" * 40)
    
    try:
        from core.attention_extractor import AttentionExtractor, AttentionStore
        
        # 创建提取器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = AttentionExtractor(device)
        
        # 创建模拟注意力图谱
        import torch
        mock_attention = torch.rand(64, 64)
        
        logger.info("生成模拟注意力图谱...")
        logger.info(f"注意力图谱形状: {mock_attention.shape}")
        
        # 生成语义掩码
        start_time = time.time()
        semantic_mask = extractor.generate_semantic_mask(mock_attention)
        end_time = time.time()
        
        logger.info(f"语义掩码生成完成，耗时: {end_time - start_time:.4f}秒")
        logger.info(f"掩码形状: {semantic_mask.shape}")
        
        # 计算前景背景比例
        foreground_ratio = (semantic_mask == 1).float().mean().item()
        logger.info(f"前景比例: {foreground_ratio:.3f}")
        
        return semantic_mask
        
    except Exception as e:
        logger.error(f"注意力提取测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主演示函数"""
    logger = get_default_logger()
    
    logger.info("=" * 60)
    logger.info("SAAH-WM Baseline 第一步完整演示")
    logger.info("=" * 60)
    
    # 确保输出目录存在
    ensure_dir("outputs")
    ensure_dir("logs")
    
    # 设置随机种子
    set_random_seed(42)
    logger.info("已设置随机种子为42")
    
    try:
        # 测试1：语义哈希生成（需要CLIP模型）
        logger.info("开始测试语义哈希生成...")
        c_bits = test_semantic_hash_with_models()
        
        if c_bits is None:
            logger.error("语义哈希生成失败，跳过后续测试")
            return False
        
        # 测试2：基准水印生成
        logger.info("开始测试基准水印生成...")
        w_base = test_base_watermark(c_bits)
        
        if w_base is None:
            logger.error("基准水印生成失败")
            return False
        
        # 测试3：信息包生成
        logger.info("开始测试信息包生成...")
        packet = test_message_packet(c_bits)
        
        if packet is None:
            logger.error("信息包生成失败")
            return False
        
        # 测试4：注意力提取（简化版）
        logger.info("开始测试注意力提取...")
        semantic_mask = test_attention_extraction_simple()
        
        if semantic_mask is None:
            logger.error("注意力提取失败")
            return False
        
        # 总结
        logger.info("=" * 60)
        logger.info("完整演示执行成功！")
        logger.info("=" * 60)
        logger.info("生成的核心信息:")
        logger.info(f"1. 语义哈希 (c_bits): {len(c_bits)}位")
        logger.info(f"2. 基准水印 (W_base): {w_base.shape}")
        logger.info(f"3. 语义掩码 (M_sem_binary): {semantic_mask.shape}")
        logger.info(f"4. 信息包 (M): {len(packet)}位")
        logger.info("=" * 60)
        logger.info("🎉 SAAH-WM Baseline 第一步实现完成！")
        
        return True
        
    except Exception as e:
        logger.error(f"演示执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断程序")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)
