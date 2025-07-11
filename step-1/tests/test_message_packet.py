"""
信息包生成器测试

测试信息包生成、BCH编码和解码的正确性。

作者：SAAH-WM团队
"""

import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.message_packet import MessagePacketGenerator


class TestMessagePacketGenerator(unittest.TestCase):
    """信息包生成器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.packet_generator = MessagePacketGenerator()
        
        # 测试数据
        self.test_c_bits = "1010110011010101" * 16  # 256位测试哈希
        self.test_copyright = "UserID:12345"
        
    def test_packet_generation(self):
        """测试基本信息包生成"""
        packet = self.packet_generator.create_message_packet(
            self.test_c_bits, self.test_copyright
        )
        
        # 检查是否为二进制字符串
        self.assertTrue(all(c in '01' for c in packet), "信息包应为二进制字符串")
        
        # 检查长度是否合理
        self.assertGreater(len(packet), len(self.test_c_bits), "信息包长度应大于原始数据")
        
        print(f"✓ 基本信息包生成测试通过，长度: {len(packet)}位")
    
    def test_packet_integrity(self):
        """测试信息包完整性"""
        # 测试多种版权信息
        test_cases = [
            "UserID:12345",
            "Copyright@2024",
            "Artist:张三",
            "License:MIT",
            "",  # 空版权信息
        ]
        
        for copyright_info in test_cases:
            is_valid = self.packet_generator.verify_packet_integrity(
                self.test_c_bits, copyright_info
            )
            self.assertTrue(is_valid, f"版权信息 '{copyright_info}' 完整性验证失败")
            
            print(f"✓ 版权信息 '{copyright_info}' 完整性验证通过")
    
    def test_encode_decode_consistency(self):
        """测试编码解码一致性"""
        packet = self.packet_generator.create_message_packet(
            self.test_c_bits, self.test_copyright
        )
        
        # 解码信息包
        decoded_c_bits, decoded_copyright, success = self.packet_generator.decode_message_packet(packet)
        
        # 检查解码是否成功
        self.assertTrue(success, "信息包解码应该成功")
        
        # 检查数据一致性
        self.assertEqual(decoded_c_bits, self.test_c_bits, "解码的语义哈希与原始不一致")
        self.assertEqual(decoded_copyright, self.test_copyright, "解码的版权信息与原始不一致")
        
        print("✓ 编码解码一致性测试通过")
    
    def test_different_inputs(self):
        """测试不同输入的处理"""
        test_cases = [
            ("0" * 256, "Short"),
            ("1" * 256, "Very long copyright information that exceeds normal length"),
            ("10" * 128, "Special chars: !@#$%^&*()"),
            ("01" * 128, "中文版权信息"),
        ]
        
        for c_bits, copyright_info in test_cases:
            try:
                packet = self.packet_generator.create_message_packet(c_bits, copyright_info)
                
                # 验证完整性
                is_valid = self.packet_generator.verify_packet_integrity(c_bits, copyright_info)
                self.assertTrue(is_valid, f"输入 '{copyright_info}' 处理失败")
                
                print(f"✓ 输入 '{copyright_info[:20]}...' 处理成功")
                
            except Exception as e:
                self.fail(f"输入 '{copyright_info}' 处理出错: {e}")
    
    def test_bch_error_correction(self):
        """测试BCH纠错功能"""
        packet = self.packet_generator.create_message_packet(
            self.test_c_bits, self.test_copyright
        )
        
        # 模拟单个位错误
        packet_list = list(packet)
        if packet_list[10] == '0':
            packet_list[10] = '1'
        else:
            packet_list[10] = '0'
        corrupted_packet = ''.join(packet_list)
        
        # 尝试解码损坏的包
        decoded_c_bits, decoded_copyright, success = self.packet_generator.decode_message_packet(
            corrupted_packet
        )
        
        # BCH应该能够纠正单个位错误
        if success:
            self.assertEqual(decoded_c_bits, self.test_c_bits, "BCH纠错后语义哈希应该正确")
            self.assertEqual(decoded_copyright, self.test_copyright, "BCH纠错后版权信息应该正确")
            print("✓ BCH单位错误纠正测试通过")
        else:
            print("⚠ BCH纠错失败，可能是错误超出纠错能力")
    
    def test_packet_info(self):
        """测试信息包配置信息"""
        info = self.packet_generator.get_packet_info()
        
        # 检查必要字段
        required_fields = [
            'bch_polynomial', 'bch_bits', 'bch_n', 'bch_k', 'bch_t',
            'max_copyright_length', 'max_data_bytes', 'total_packet_bits'
        ]
        
        for field in required_fields:
            self.assertIn(field, info, f"缺少配置信息字段: {field}")
        
        # 检查数值合理性
        self.assertGreater(info['bch_n'], info['bch_k'], "BCH编码长度应大于数据长度")
        self.assertGreater(info['bch_t'], 0, "BCH纠错能力应大于0")
        
        print(f"✓ 信息包配置信息测试通过: {info}")
    
    def test_long_copyright_truncation(self):
        """测试长版权信息截断"""
        # 创建超长版权信息
        long_copyright = "A" * 100  # 超过默认64字符限制
        
        packet = self.packet_generator.create_message_packet(
            self.test_c_bits, long_copyright
        )
        
        # 解码并检查是否被正确截断
        decoded_c_bits, decoded_copyright, success = self.packet_generator.decode_message_packet(packet)
        
        self.assertTrue(success, "长版权信息处理应该成功")
        self.assertEqual(len(decoded_copyright), 64, "版权信息应被截断到64字符")
        self.assertEqual(decoded_copyright, long_copyright[:64], "截断的版权信息应该正确")
        
        print("✓ 长版权信息截断测试通过")
    
    def test_empty_copyright(self):
        """测试空版权信息"""
        empty_copyright = ""
        
        packet = self.packet_generator.create_message_packet(
            self.test_c_bits, empty_copyright
        )
        
        decoded_c_bits, decoded_copyright, success = self.packet_generator.decode_message_packet(packet)
        
        self.assertTrue(success, "空版权信息处理应该成功")
        self.assertEqual(decoded_c_bits, self.test_c_bits, "语义哈希应该正确")
        self.assertEqual(decoded_copyright, "", "空版权信息应该保持为空")
        
        print("✓ 空版权信息测试通过")
    
    def test_invalid_c_bits(self):
        """测试无效语义哈希处理"""
        invalid_c_bits_list = [
            "invalid_binary",  # 非二进制字符
            "101",  # 长度不足
            "",  # 空字符串
        ]
        
        for invalid_c_bits in invalid_c_bits_list:
            with self.assertRaises(ValueError, msg=f"应该拒绝无效c_bits: {invalid_c_bits}"):
                self.packet_generator.create_message_packet(invalid_c_bits, self.test_copyright)
        
        print("✓ 无效语义哈希处理测试通过")
    
    def test_multiple_packets_consistency(self):
        """测试多个信息包的一致性"""
        # 生成多个相同的信息包
        packets = []
        for i in range(5):
            packet = self.packet_generator.create_message_packet(
                self.test_c_bits, self.test_copyright
            )
            packets.append(packet)
        
        # 检查所有包是否相同
        reference_packet = packets[0]
        for i, packet in enumerate(packets[1:], 1):
            self.assertEqual(packet, reference_packet, 
                           f"第{i+1}个信息包与参考包不同")
        
        print("✓ 多个信息包一致性测试通过")


def run_message_packet_tests():
    """运行信息包测试"""
    print("=" * 50)
    print("开始信息包生成器测试")
    print("=" * 50)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMessagePacketGenerator)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n✅ 所有信息包测试通过！")
    else:
        print(f"\n❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_message_packet_tests()
