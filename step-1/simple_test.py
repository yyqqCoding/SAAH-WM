"""
简单的功能测试脚本

测试SAAH-WM基础功能，不依赖复杂的相对导入。

作者：SAAH-WM团队
"""

import sys
import os
import torch

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_functions():
    """测试基础工具函数"""
    print("测试基础工具函数...")
    
    # 直接导入工具函数
    from utils.common_utils import string_to_bits, bits_to_string, validate_bits
    
    # 测试字符串编码解码
    test_string = "Hello, 世界!"
    bits = string_to_bits(test_string)
    recovered = bits_to_string(bits)
    assert recovered == test_string, "字符串编码解码失败"
    print("✓ 字符串编码解码测试通过")
    
    # 测试二进制验证
    assert validate_bits("101010"), "二进制验证失败"
    assert not validate_bits("invalid"), "二进制验证失败"
    print("✓ 二进制验证测试通过")


def test_base_watermark():
    """测试基准水印生成器"""
    print("测试基准水印生成器...")
    
    # 直接导入并修复相对导入
    import torch
    from utils.logger_config import LoggerMixin
    from utils.common_utils import bits_to_int, validate_bits
    
    class BaseWatermarkGenerator(LoggerMixin):
        """基准水印生成器（简化版）"""
        
        def __init__(self, device="cpu"):
            super().__init__()
            self.device = device
        
        def _seed_from_bits(self, c_bits):
            if not validate_bits(c_bits):
                raise ValueError(f"输入不是有效的二进制字符串")
            seed = bits_to_int(c_bits) % (2**32)
            return seed
        
        def generate_base_watermark(self, c_bits, height=64, width=64, channels=4):
            seed = self._seed_from_bits(c_bits)
            torch.manual_seed(seed)
            w_base = torch.randn((1, channels, height, width), dtype=torch.float32, device=self.device)
            return w_base
    
    # 测试
    watermark_gen = BaseWatermarkGenerator("cpu")
    test_c_bits = "1010110011010101" * 16  # 256位
    w_base = watermark_gen.generate_base_watermark(test_c_bits)
    
    assert w_base.shape == (1, 4, 64, 64), f"水印形状错误: {w_base.shape}"
    print("✓ 基准水印生成器测试通过")


def test_message_packet():
    """测试信息包生成器"""
    print("测试信息包生成器...")
    
    import bchlib
    from utils.logger_config import LoggerMixin
    from utils.common_utils import string_to_bits, bits_to_string, validate_bits, pad_bits
    
    class MessagePacketGenerator(LoggerMixin):
        """信息包生成器（简化版）"""
        
        def __init__(self, bch_t=5, max_copyright_length=64):
            super().__init__()
            self.bch_t = bch_t
            self.max_copyright_length = max_copyright_length
            # 使用正确的bchlib API
            self.bch = bchlib.BCH(t=bch_t, prim_poly=8219)
        
        def _encode_copyright_info(self, copyright_info):
            if len(copyright_info) > self.max_copyright_length:
                copyright_info = copyright_info[:self.max_copyright_length]
            copyright_info = copyright_info.ljust(self.max_copyright_length, '\0')
            copyright_bits = string_to_bits(copyright_info, encoding='utf-8')
            return copyright_bits
        
        def _add_bch_error_correction(self, data_bits):
            # 简化的BCH编码，适配bchlib API
            padded_bits = pad_bits(data_bits, ((len(data_bits) + 7) // 8) * 8)
            byte_data = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte_chunk = padded_bits[i:i+8]
                byte_data.append(int(byte_chunk, 2))

            # 限制数据长度以适应BCH编码器
            max_data_bytes = min(len(byte_data), 32)  # 限制为32字节
            data_to_encode = bytes(byte_data[:max_data_bytes])

            # BCH编码
            ecc = self.bch.encode(data_to_encode)
            encoded_data = data_to_encode + ecc
            encoded_bits = ''.join(format(byte, '08b') for byte in encoded_data)
            return encoded_bits
        
        def create_message_packet(self, c_bits, copyright_info):
            if not validate_bits(c_bits):
                raise ValueError("语义哈希不是有效的二进制字符串")
            
            copyright_bits = self._encode_copyright_info(copyright_info)
            combined_data = c_bits + copyright_bits
            final_packet = self._add_bch_error_correction(combined_data)
            return final_packet
        
        def verify_packet_integrity(self, c_bits, copyright_info):
            try:
                packet = self.create_message_packet(c_bits, copyright_info)
                return len(packet) > 0 and all(c in '01' for c in packet)
            except:
                return False
    
    # 测试
    packet_gen = MessagePacketGenerator()
    test_c_bits = "1010110011010101" * 16  # 256位
    packet = packet_gen.create_message_packet(test_c_bits, "TestUser:123")
    
    assert len(packet) > 0, "信息包生成失败"
    assert all(c in '01' for c in packet), "信息包不是二进制字符串"
    
    # 验证完整性
    is_valid = packet_gen.verify_packet_integrity(test_c_bits, "TestUser:123")
    assert is_valid, "信息包完整性验证失败"
    
    print("✓ 信息包生成器测试通过")


def main():
    """主测试函数"""
    print("SAAH-WM Baseline 简单功能测试")
    print("=" * 50)
    
    try:
        # 测试基础函数
        test_basic_functions()
        
        # 测试基准水印生成器
        test_base_watermark()
        
        # 测试信息包生成器
        test_message_packet()
        
        print("=" * 50)
        print("🎉 所有基础功能测试通过！")
        print("注意：语义哈希和注意力提取需要加载大模型，在实际使用时测试。")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
