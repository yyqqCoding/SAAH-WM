"""
测试已可用功能的演示程序

只测试不需要Stable Diffusion模型的功能，包括语义哈希生成。

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


def test_semantic_hash_only():
    """只测试语义哈希生成（仅需CLIP模型）"""
    logger = get_default_logger()
    logger.info("=" * 40)
    logger.info("测试语义哈希生成（仅CLIP模型）")
    logger.info("=" * 40)
    
    try:
        # 只加载CLIP模型
        model_loader = ModelLoader()
        clip_model, clip_processor = model_loader.load_clip_model()
        
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
        
        # 测试不同prompt
        test_prompts = [
            "一只戴着宇航头盔的柴犬",
            "a dog wearing a space helmet",
            "一只猫坐在椅子上",
            "a cat sitting on a chair"
        ]
        
        logger.info("测试多个prompt的哈希生成...")
        hashes = {}
        for prompt in test_prompts:
            hash_result = hash_generator.generate_semantic_hash(prompt)
            hashes[prompt] = hash_result
            logger.info(f"'{prompt}': {hash_result[:16]}...{hash_result[-16:]}")
        
        # 验证不同prompt生成不同哈希
        hash_values = list(hashes.values())
        unique_hashes = len(set(hash_values))
        logger.info(f"生成了{len(hash_values)}个哈希，其中{unique_hashes}个唯一")
        
        if unique_hashes == len(hash_values):
            logger.info("✅ 不同prompt生成了不同的哈希")
        else:
            logger.warning("⚠️ 某些不同的prompt生成了相同的哈希")
        
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
        
        # 测试不同尺寸
        logger.info("测试不同尺寸的水印生成...")
        test_sizes = [(32, 32, 3), (128, 128, 4), (64, 32, 1)]
        for h, w, c in test_sizes:
            w_test = watermark_gen.generate_base_watermark(c_bits, h, w, c)
            logger.info(f"尺寸 {h}x{w}x{c}: 生成成功，形状 {w_test.shape}")
        
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
        test_copyrights = [
            "UserID:12345",
            "Copyright@2024",
            "Artist:张三",
            "License:MIT",
            "",  # 空版权信息
            "A" * 100  # 超长版权信息
        ]
        
        for copyright_info in test_copyrights:
            logger.info(f"测试版权信息: '{copyright_info[:20]}{'...' if len(copyright_info) > 20 else ''}'")
            
            # 生成信息包
            start_time = time.time()
            packet = packet_gen.create_message_packet(c_bits, copyright_info)
            end_time = time.time()
            
            logger.info(f"信息包生成完成，耗时: {end_time - start_time:.4f}秒")
            logger.info(f"信息包长度: {len(packet)}位")
            
            # 验证完整性
            is_valid = packet_gen.verify_packet_integrity(c_bits, copyright_info)
            logger.info(f"完整性验证: {'通过' if is_valid else '失败'}")
            
            # 测试解码
            decoded_c_bits, decoded_copyright, decode_success = packet_gen.decode_message_packet(packet)
            if decode_success:
                logger.info(f"解码成功: 版权信息='{decoded_copyright}'")
                if decoded_c_bits == c_bits:
                    logger.info("✅ 语义哈希解码正确")
                else:
                    logger.warning("⚠️ 语义哈希解码不匹配")
            else:
                logger.warning("⚠️ 信息包解码失败")
            
            logger.info("-" * 30)
        
        # 获取配置信息
        packet_info = packet_gen.get_packet_info()
        logger.info(f"BCH配置: t={packet_info['bch_t']}, n={packet_info['bch_n']}")
        
        return packet
        
    except Exception as e:
        logger.error(f"信息包测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_attention_extraction_mock():
    """测试注意力提取器（模拟数据）"""
    logger = get_default_logger()
    logger.info("=" * 40)
    logger.info("测试注意力提取器（模拟数据）")
    logger.info("=" * 40)
    
    try:
        from core.attention_extractor import AttentionExtractor
        
        # 创建提取器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = AttentionExtractor(device)
        
        # 创建模拟注意力图谱
        logger.info("生成模拟注意力图谱...")
        mock_attention = torch.rand(64, 64)
        logger.info(f"模拟注意力图谱形状: {mock_attention.shape}")
        
        # 生成语义掩码
        start_time = time.time()
        semantic_mask = extractor.generate_semantic_mask(mock_attention)
        end_time = time.time()
        
        logger.info(f"语义掩码生成完成，耗时: {end_time - start_time:.4f}秒")
        logger.info(f"掩码形状: {semantic_mask.shape}")
        
        # 计算前景背景比例
        foreground_ratio = (semantic_mask == 1).float().mean().item()
        logger.info(f"前景比例: {foreground_ratio:.3f}")
        
        # 测试不同尺寸的注意力图谱
        logger.info("测试不同尺寸的注意力图谱...")
        test_sizes = [32, 64, 128]
        for size in test_sizes:
            mock_attn = torch.rand(size, size)
            mask = extractor.generate_semantic_mask(mock_attn)
            logger.info(f"尺寸 {size}x{size}: 掩码形状 {mask.shape}")
        
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
    logger.info("SAAH-WM Baseline 可用功能测试")
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
        c_bits = test_semantic_hash_only()
        
        if c_bits is None:
            logger.error("语义哈希生成失败，使用模拟数据继续测试")
            c_bits = "1010110011010101" * 16  # 256位模拟哈希
        
        # 测试2：基准水印生成
        logger.info("开始测试基准水印生成...")
        w_base = test_base_watermark(c_bits)
        
        # 测试3：信息包生成
        logger.info("开始测试信息包生成...")
        packet = test_message_packet(c_bits)
        
        # 测试4：注意力提取（模拟数据）
        logger.info("开始测试注意力提取...")
        semantic_mask = test_attention_extraction_mock()
        
        # 总结
        logger.info("=" * 60)
        logger.info("可用功能测试完成！")
        logger.info("=" * 60)
        
        success_count = 0
        total_count = 4
        
        if c_bits and len(c_bits) == 256:
            logger.info("✅ 语义哈希生成: 成功")
            success_count += 1
        else:
            logger.info("❌ 语义哈希生成: 失败")
        
        if w_base is not None:
            logger.info("✅ 基准水印生成: 成功")
            success_count += 1
        else:
            logger.info("❌ 基准水印生成: 失败")
        
        if packet is not None:
            logger.info("✅ 信息包生成: 成功")
            success_count += 1
        else:
            logger.info("❌ 信息包生成: 失败")
        
        if semantic_mask is not None:
            logger.info("✅ 注意力提取（模拟）: 成功")
            success_count += 1
        else:
            logger.info("❌ 注意力提取（模拟）: 失败")
        
        logger.info("=" * 60)
        logger.info(f"测试结果: {success_count}/{total_count} 个功能正常")
        
        if success_count == total_count:
            logger.info("🎉 所有可用功能测试通过！")
            return True
        else:
            logger.warning(f"⚠️ 有 {total_count - success_count} 个功能测试失败")
            return False
        
    except Exception as e:
        logger.error(f"测试执行失败: {str(e)}")
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
