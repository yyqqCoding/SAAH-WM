"""
基准水印生成器测试

测试基准水印生成的确定性、一致性和正确性。

作者：SAAH-WM团队
"""

import unittest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_watermark import BaseWatermarkGenerator


class TestBaseWatermarkGenerator(unittest.TestCase):
    """基准水印生成器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.watermark_generator = BaseWatermarkGenerator(self.device)
        
        # 测试用的语义哈希
        self.test_c_bits = "1010110011010101" * 16  # 256位测试哈希
        
    def test_watermark_generation(self):
        """测试基本水印生成功能"""
        w_base = self.watermark_generator.generate_base_watermark(self.test_c_bits)
        
        # 检查形状
        expected_shape = (1, 4, 64, 64)
        self.assertEqual(w_base.shape, expected_shape, f"水印形状应为{expected_shape}")
        
        # 检查数据类型
        self.assertEqual(w_base.dtype, torch.float32, "水印数据类型应为float32")
        
        # 检查设备
        self.assertEqual(str(w_base.device), self.device, f"水印应在设备{self.device}上")
        
        # 检查数值范围（正态分布，大部分值应在[-3, 3]范围内）
        self.assertTrue(torch.all(torch.abs(w_base) < 10), "水印值应在合理范围内")
        
        print(f"✓ 基本水印生成测试通过，形状: {w_base.shape}")
    
    def test_deterministic_generation(self):
        """测试确定性生成"""
        # 使用相同的c_bits多次生成水印
        watermarks = []
        for i in range(5):
            w_base = self.watermark_generator.generate_base_watermark(self.test_c_bits)
            watermarks.append(w_base)
        
        # 检查所有水印是否完全相同
        reference = watermarks[0]
        for i, watermark in enumerate(watermarks[1:], 1):
            self.assertTrue(torch.equal(reference, watermark), 
                          f"第{i+1}次生成的水印与参考水印不同")
        
        print("✓ 确定性生成测试通过")
    
    def test_different_seeds_different_watermarks(self):
        """测试不同种子生成不同水印"""
        # 创建不同的c_bits
        c_bits_list = [
            "1010110011010101" * 16,  # 原始
            "0101001100101010" * 16,  # 反转
            "1111000011110000" * 16,  # 模式1
            "0000111100001111" * 16,  # 模式2
        ]
        
        watermarks = []
        for c_bits in c_bits_list:
            w_base = self.watermark_generator.generate_base_watermark(c_bits)
            watermarks.append(w_base)
        
        # 检查所有水印都不相同
        for i in range(len(watermarks)):
            for j in range(i + 1, len(watermarks)):
                self.assertFalse(torch.equal(watermarks[i], watermarks[j]), 
                               f"不同c_bits生成了相同的水印 (索引{i}, {j})")
        
        print("✓ 不同种子生成不同水印测试通过")
    
    def test_custom_dimensions(self):
        """测试自定义尺寸"""
        test_cases = [
            (32, 32, 3),   # 较小尺寸
            (128, 128, 4), # 较大尺寸
            (64, 32, 1),   # 非方形
        ]
        
        for height, width, channels in test_cases:
            w_base = self.watermark_generator.generate_base_watermark(
                self.test_c_bits, height, width, channels
            )
            
            expected_shape = (1, channels, height, width)
            self.assertEqual(w_base.shape, expected_shape, 
                           f"自定义尺寸{expected_shape}测试失败")
            
            print(f"✓ 自定义尺寸{expected_shape}测试通过")
    
    def test_batch_generation(self):
        """测试批量生成"""
        c_bits_list = [
            "1010110011010101" * 16,
            "0101001100101010" * 16,
            "1111000011110000" * 16,
        ]
        
        batch_watermarks = self.watermark_generator.generate_watermark_batch(c_bits_list)
        
        # 检查批次形状
        expected_shape = (3, 4, 64, 64)
        self.assertEqual(batch_watermarks.shape, expected_shape, 
                        f"批次水印形状应为{expected_shape}")
        
        # 检查每个水印的一致性
        for i, c_bits in enumerate(c_bits_list):
            individual_watermark = self.watermark_generator.generate_base_watermark(c_bits)
            batch_watermark = batch_watermarks[i:i+1]
            
            self.assertTrue(torch.equal(individual_watermark, batch_watermark),
                          f"批次中第{i}个水印与单独生成的不一致")
        
        print("✓ 批量生成测试通过")
    
    def test_watermark_statistics(self):
        """测试水印统计信息"""
        w_base = self.watermark_generator.generate_base_watermark(self.test_c_bits)
        stats = self.watermark_generator.get_watermark_statistics(w_base)
        
        # 检查统计信息字段
        required_fields = ['shape', 'dtype', 'device', 'mean', 'std', 'min', 'max', 'num_elements']
        for field in required_fields:
            self.assertIn(field, stats, f"缺少统计信息字段: {field}")
        
        # 检查数值的合理性
        self.assertEqual(stats['num_elements'], 64 * 64 * 4, "元素数量计算错误")
        self.assertGreater(stats['std'], 0, "标准差应大于0")
        
        # 对于正态分布，均值应接近0
        self.assertLess(abs(stats['mean']), 0.5, "均值应接近0")
        
        print(f"✓ 水印统计信息测试通过: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    
    def test_watermark_comparison(self):
        """测试水印比较功能"""
        # 生成两个相同的水印
        w1 = self.watermark_generator.generate_base_watermark(self.test_c_bits)
        w2 = self.watermark_generator.generate_base_watermark(self.test_c_bits)
        
        # 比较相同水印
        stats_same = self.watermark_generator.compare_watermarks(w1, w2)
        self.assertTrue(stats_same['are_equal'], "相同种子生成的水印应该相等")
        self.assertEqual(stats_same['mse'], 0.0, "相同水印的MSE应为0")
        self.assertEqual(stats_same['correlation'], 1.0, "相同水印的相关系数应为1")
        
        # 生成不同的水印
        different_c_bits = "0101001100101010" * 16
        w3 = self.watermark_generator.generate_base_watermark(different_c_bits)
        
        # 比较不同水印
        stats_diff = self.watermark_generator.compare_watermarks(w1, w3)
        self.assertFalse(stats_diff['are_equal'], "不同种子生成的水印应该不相等")
        self.assertGreater(stats_diff['mse'], 0, "不同水印的MSE应大于0")
        self.assertLess(abs(stats_diff['correlation']), 0.1, "不同水印的相关系数应接近0")
        
        print("✓ 水印比较功能测试通过")
    
    def test_invalid_input(self):
        """测试无效输入处理"""
        # 测试无效的二进制字符串
        invalid_c_bits = [
            "invalid_binary",  # 包含非二进制字符
            "101",  # 长度不足
            "",  # 空字符串
        ]
        
        for invalid_bits in invalid_c_bits:
            with self.assertRaises(ValueError, msg=f"应该拒绝无效输入: {invalid_bits}"):
                self.watermark_generator.generate_base_watermark(invalid_bits)
        
        print("✓ 无效输入处理测试通过")


def run_base_watermark_tests():
    """运行基准水印测试"""
    print("=" * 50)
    print("开始基准水印生成器测试")
    print("=" * 50)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseWatermarkGenerator)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n✅ 所有基准水印测试通过！")
    else:
        print(f"\n❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_base_watermark_tests()
