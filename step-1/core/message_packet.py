"""
信息包生成模块

将语义哈希和版权信息组合，使用BCH纠错码生成最终的信息包。
参考EditGuard的实现方式进行BCH编码。

作者：SAAH-WM团队
"""

import bchlib
import hashlib
from typing import Optional, Tuple

from ..utils.logger_config import LoggerMixin
from ..utils.common_utils import string_to_bits, bits_to_string, validate_bits, pad_bits


class MessagePacketGenerator(LoggerMixin):
    """
    信息包生成器
    
    负责将语义哈希c_bits和版权信息组合，
    并使用BCH纠错码生成最终的信息包。
    """
    
    def __init__(
        self,
        bch_t: int = 5,
        bch_prim_poly: int = 8219,
        max_copyright_length: int = 64
    ):
        """
        初始化信息包生成器

        Args:
            bch_t: BCH纠错能力（可纠正的错误位数）
            bch_prim_poly: BCH原始多项式
            max_copyright_length: 版权信息最大长度（字符）
        """
        super().__init__()

        self.bch_t = bch_t
        self.bch_prim_poly = bch_prim_poly
        self.max_copyright_length = max_copyright_length

        # 初始化BCH编码器
        try:
            self.bch = bchlib.BCH(t=bch_t, prim_poly=bch_prim_poly)
            self.log_info(f"BCH编码器初始化成功，纠错能力: {bch_t}, 多项式: {bch_prim_poly}")
            self.log_info(f"BCH参数 - n: {self.bch.n}, t: {self.bch.t}")
        except Exception as e:
            self.log_error(f"BCH编码器初始化失败: {str(e)}")
            raise

        self.log_info(f"信息包生成器初始化完成，最大版权信息长度: {max_copyright_length}字符")
    
    def _encode_copyright_info(self, copyright_info: str) -> str:
        """
        将版权信息编码为二进制字符串
        
        Args:
            copyright_info: 版权信息字符串
            
        Returns:
            二进制编码的版权信息
        """
        if len(copyright_info) > self.max_copyright_length:
            self.log_warning(f"版权信息长度超过限制，将截断到{self.max_copyright_length}字符")
            copyright_info = copyright_info[:self.max_copyright_length]
        
        # 填充到固定长度
        copyright_info = copyright_info.ljust(self.max_copyright_length, '\0')
        
        # 转换为二进制
        copyright_bits = string_to_bits(copyright_info, encoding='utf-8')
        
        self.log_debug(f"版权信息编码完成，长度: {len(copyright_bits)}位")
        return copyright_bits
    
    def _add_bch_error_correction(self, data_bits: str) -> str:
        """
        为数据添加BCH纠错码
        
        Args:
            data_bits: 原始数据的二进制字符串
            
        Returns:
            添加纠错码后的二进制字符串
        """
        try:
            # 将二进制字符串转换为字节
            # 确保位数是8的倍数
            padded_bits = pad_bits(data_bits, ((len(data_bits) + 7) // 8) * 8)
            
            # 转换为字节数组
            byte_data = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte_chunk = padded_bits[i:i+8]
                byte_data.append(int(byte_chunk, 2))
            
            self.log_debug(f"原始数据长度: {len(byte_data)}字节")
            
            # 确保数据长度不超过BCH编码器的限制
            # bchlib没有k属性，使用固定的最大数据长度
            max_data_bytes = min(len(byte_data), 32)  # 限制为32字节
            if len(byte_data) > max_data_bytes:
                self.log_warning(f"数据长度超过BCH限制，截断到{max_data_bytes}字节")
                byte_data = byte_data[:max_data_bytes]
            
            # 计算BCH纠错码
            data_to_encode = bytes(byte_data)
            ecc = self.bch.encode(data_to_encode)

            # 组合原始数据和纠错码
            encoded_data = data_to_encode + ecc
            
            # 转换回二进制字符串
            encoded_bits = ''.join(format(byte, '08b') for byte in encoded_data)
            
            self.log_info(f"BCH编码完成，原始长度: {len(data_bits)}位, "
                         f"编码后长度: {len(encoded_bits)}位")
            
            return encoded_bits
            
        except Exception as e:
            self.log_error(f"BCH编码失败: {str(e)}")
            raise
    
    def create_message_packet(self, c_bits: str, copyright_info: str) -> str:
        """
        创建完整的信息包
        
        Args:
            c_bits: 语义哈希二进制字符串
            copyright_info: 版权信息字符串
            
        Returns:
            最终的信息包二进制字符串
        """
        self.log_info("开始创建信息包...")
        self.log_debug(f"语义哈希长度: {len(c_bits)}位")
        self.log_debug(f"版权信息: '{copyright_info}'")
        
        try:
            # 验证输入
            if not validate_bits(c_bits):
                raise ValueError(f"语义哈希不是有效的二进制字符串")
            
            # 步骤1：编码版权信息
            self.log_debug("正在编码版权信息...")
            copyright_bits = self._encode_copyright_info(copyright_info)
            
            # 步骤2：拼接语义哈希和版权信息
            self.log_debug("正在拼接数据...")
            combined_data = c_bits + copyright_bits
            
            self.log_info(f"拼接后数据长度: {len(combined_data)}位 "
                         f"(语义哈希: {len(c_bits)}位 + 版权信息: {len(copyright_bits)}位)")
            
            # 步骤3：添加BCH纠错码
            self.log_debug("正在添加BCH纠错码...")
            final_packet = self._add_bch_error_correction(combined_data)
            
            self.log_info(f"信息包创建完成，最终长度: {len(final_packet)}位")
            
            return final_packet
            
        except Exception as e:
            self.log_error(f"信息包创建失败: {str(e)}")
            raise
    
    def decode_message_packet(self, packet_bits: str) -> Tuple[str, str, bool]:
        """
        解码信息包（用于验证）
        
        Args:
            packet_bits: 信息包二进制字符串
            
        Returns:
            (语义哈希, 版权信息, 是否解码成功)
        """
        self.log_info("开始解码信息包...")
        
        try:
            # 将二进制字符串转换为字节
            padded_bits = pad_bits(packet_bits, ((len(packet_bits) + 7) // 8) * 8)
            
            byte_data = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte_chunk = padded_bits[i:i+8]
                byte_data.append(int(byte_chunk, 2))
            
            # BCH解码
            # bchlib的解码需要分离数据和ECC部分
            ecc_bytes = self.bch.ecc_bytes

            if len(byte_data) < ecc_bytes:
                self.log_error(f"数据长度不足，需要至少{ecc_bytes}字节，实际{len(byte_data)}字节")
                return "", "", False

            # 分离数据和纠错码
            data_bytes = len(byte_data) - ecc_bytes
            data_part = bytes(byte_data[:data_bytes])
            ecc_part = bytes(byte_data[data_bytes:data_bytes + ecc_bytes])
            
            # 尝试纠错
            try:
                corrected_data, num_errors = self.bch.decode(data_part, ecc_part)
                self.log_info(f"BCH解码成功，纠正了{num_errors}个错误")
                decode_success = True
            except Exception as decode_error:
                self.log_warning(f"BCH解码失败: {str(decode_error)}")
                corrected_data = data_part
                decode_success = False
            
            # 转换回二进制字符串
            corrected_bits = ''.join(format(byte, '08b') for byte in corrected_data)
            
            # 分离语义哈希和版权信息
            # 假设语义哈希长度为256位
            hash_length = 256
            c_bits_decoded = corrected_bits[:hash_length]
            copyright_bits = corrected_bits[hash_length:]
            
            # 解码版权信息
            try:
                copyright_info = bits_to_string(copyright_bits, encoding='utf-8')
                # 移除填充的空字符
                copyright_info = copyright_info.rstrip('\0')
            except Exception as e:
                self.log_warning(f"版权信息解码失败: {str(e)}")
                copyright_info = ""
                decode_success = False
            
            self.log_info(f"信息包解码完成，成功: {decode_success}")
            return c_bits_decoded, copyright_info, decode_success
            
        except Exception as e:
            self.log_error(f"信息包解码失败: {str(e)}")
            return "", "", False
    
    def verify_packet_integrity(self, c_bits: str, copyright_info: str) -> bool:
        """
        验证信息包的完整性（编码后再解码验证）
        
        Args:
            c_bits: 语义哈希
            copyright_info: 版权信息
            
        Returns:
            是否验证成功
        """
        self.log_info("开始验证信息包完整性...")
        
        try:
            # 创建信息包
            packet = self.create_message_packet(c_bits, copyright_info)
            
            # 解码验证
            decoded_c_bits, decoded_copyright, success = self.decode_message_packet(packet)
            
            # 检查是否一致
            c_bits_match = decoded_c_bits == c_bits
            copyright_match = decoded_copyright == copyright_info
            
            overall_success = success and c_bits_match and copyright_match
            
            self.log_info(f"完整性验证结果: {overall_success}")
            self.log_debug(f"BCH解码成功: {success}")
            self.log_debug(f"语义哈希匹配: {c_bits_match}")
            self.log_debug(f"版权信息匹配: {copyright_match}")
            
            return overall_success
            
        except Exception as e:
            self.log_error(f"完整性验证失败: {str(e)}")
            return False
    
    def get_packet_info(self) -> dict:
        """
        获取信息包生成器的配置信息
        
        Returns:
            配置信息字典
        """
        return {
            "bch_t": self.bch_t,
            "bch_prim_poly": self.bch_prim_poly,
            "bch_n": self.bch.n,
            "bch_t_actual": self.bch.t,
            "bch_ecc_bytes": self.bch.ecc_bytes,
            "max_copyright_length": self.max_copyright_length,
            "max_data_bytes": 32,  # 固定限制
            "total_packet_bits": self.bch.n
        }
