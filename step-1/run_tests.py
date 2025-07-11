"""
SAAH-WM Baseline 测试运行器

运行所有模块的测试，验证系统功能的正确性。

作者：SAAH-WM团队
"""

import sys
import os
import time
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger_config import get_default_logger
from utils.common_utils import ensure_dir


def run_individual_tests() -> List[Tuple[str, bool]]:
    """
    运行各个模块的单独测试
    
    Returns:
        测试结果列表 [(模块名, 是否成功)]
    """
    logger = get_default_logger()
    results = []
    
    # 测试模块列表
    test_modules = [
        ("基准水印生成器", "tests.test_base_watermark", "run_base_watermark_tests"),
        ("信息包生成器", "tests.test_message_packet", "run_message_packet_tests"),
        ("语义哈希生成器", "tests.test_semantic_hash", "run_semantic_hash_tests"),
    ]
    
    for module_name, module_path, test_function in test_modules:
        logger.info(f"开始测试: {module_name}")
        start_time = time.time()
        
        try:
            # 动态导入测试模块
            test_module = __import__(module_path, fromlist=[test_function])
            test_func = getattr(test_module, test_function)
            
            # 运行测试
            success = test_func()
            
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                logger.info(f"✅ {module_name} 测试通过 (耗时: {duration:.2f}秒)")
            else:
                logger.error(f"❌ {module_name} 测试失败 (耗时: {duration:.2f}秒)")
            
            results.append((module_name, success))
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"❌ {module_name} 测试出错: {str(e)} (耗时: {duration:.2f}秒)")
            results.append((module_name, False))
        
        logger.info("-" * 50)
    
    return results


def run_quick_integration_test() -> bool:
    """
    运行快速集成测试（不需要加载大模型）
    
    Returns:
        是否成功
    """
    logger = get_default_logger()
    logger.info("开始快速集成测试...")
    
    try:
        # 测试基准水印生成器
        from core.base_watermark import BaseWatermarkGenerator
        
        watermark_gen = BaseWatermarkGenerator("cpu")
        test_c_bits = "1010110011010101" * 16
        w_base = watermark_gen.generate_base_watermark(test_c_bits)
        
        assert w_base.shape == (1, 4, 64, 64), "基准水印形状错误"
        logger.info("✓ 基准水印生成器集成测试通过")
        
        # 测试信息包生成器
        from core.message_packet import MessagePacketGenerator
        
        packet_gen = MessagePacketGenerator()
        packet = packet_gen.create_message_packet(test_c_bits, "TestUser:123")
        
        assert len(packet) > 0, "信息包生成失败"
        assert all(c in '01' for c in packet), "信息包不是二进制字符串"
        
        # 验证完整性
        is_valid = packet_gen.verify_packet_integrity(test_c_bits, "TestUser:123")
        assert is_valid, "信息包完整性验证失败"
        
        logger.info("✓ 信息包生成器集成测试通过")
        
        # 测试工具函数
        from utils.common_utils import string_to_bits, bits_to_string, validate_bits
        
        test_string = "Hello, 世界!"
        bits = string_to_bits(test_string)
        recovered_string = bits_to_string(bits)
        
        assert recovered_string == test_string, "字符串编码解码失败"
        assert validate_bits(bits), "二进制字符串验证失败"
        
        logger.info("✓ 工具函数集成测试通过")
        
        logger.info("✅ 快速集成测试全部通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 快速集成测试失败: {str(e)}")
        return False


def print_test_summary(results: List[Tuple[str, bool]], integration_success: bool):
    """
    打印测试总结
    
    Args:
        results: 各模块测试结果
        integration_success: 集成测试结果
    """
    logger = get_default_logger()
    
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    # 统计结果
    total_tests = len(results) + 1  # 包括集成测试
    passed_tests = sum(1 for _, success in results if success)
    if integration_success:
        passed_tests += 1
    
    # 打印各模块结果
    for module_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{module_name}: {status}")
    
    # 打印集成测试结果
    integration_status = "✅ 通过" if integration_success else "❌ 失败"
    logger.info(f"快速集成测试: {integration_status}")
    
    logger.info("-" * 40)
    logger.info(f"总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有测试都通过了！系统功能正常。")
        return True
    else:
        failed_count = total_tests - passed_tests
        logger.error(f"⚠️  有 {failed_count} 个测试失败，请检查相关模块。")
        return False


def main():
    """主函数"""
    print("SAAH-WM Baseline 第一步测试运行器")
    print("=" * 60)
    
    # 确保必要目录存在
    ensure_dir("logs")
    ensure_dir("outputs")
    
    logger = get_default_logger()
    logger.info("开始运行SAAH-WM Baseline测试套件")
    
    start_time = time.time()
    
    try:
        # 运行快速集成测试
        integration_success = run_quick_integration_test()
        
        # 运行各模块测试
        results = run_individual_tests()
        
        # 打印总结
        all_success = print_test_summary(results, integration_success)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info(f"测试套件运行完成，总耗时: {total_duration:.2f}秒")
        
        # 返回适当的退出码
        sys.exit(0 if all_success else 1)
        
    except KeyboardInterrupt:
        logger.info("用户中断测试")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试运行器出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
