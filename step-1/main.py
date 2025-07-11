"""
SAAH-WM Baseline 第一步主程序

演示四个核心模块的功能：
1. 语义哈希生成
2. 基准水印生成
3. 注意力图谱提取与掩码生成
4. 信息包生成

作者：SAAH-WM团队
"""

import os
import torch
from typing import Optional

from utils.model_loader import ModelLoader
from utils.logger_config import get_default_logger
from utils.common_utils import ensure_dir, set_random_seed

from core.semantic_hash import SemanticHashGenerator
from core.base_watermark import BaseWatermarkGenerator
from core.attention_extractor import AttentionExtractor
from core.message_packet import MessagePacketGenerator


class SAHWMBaseline:
    """
    SAAH-WM Baseline 第一步实现
    
    整合四个核心模块，提供完整的演示功能。
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化SAAH-WM Baseline系统
        
        Args:
            device: 计算设备，如果为None则自动选择
        """
        # 设置日志
        self.logger = get_default_logger()
        self.logger.info("=" * 60)
        self.logger.info("SAAH-WM Baseline 第一步系统启动")
        self.logger.info("=" * 60)
        
        # 设置随机种子
        set_random_seed(42)
        self.logger.info("已设置随机种子为42")
        
        # 初始化模型加载器
        self.model_loader = ModelLoader(device)
        self.device = self.model_loader.device
        
        # 初始化各个模块（延迟加载）
        self.semantic_hash_generator = None
        self.base_watermark_generator = None
        self.attention_extractor = None
        self.message_packet_generator = None
        
        # 模型存储
        self.clip_model = None
        self.clip_processor = None
        self.sd_pipeline = None
        
        self.logger.info(f"系统初始化完成，使用设备: {self.device}")
    
    def load_models(self):
        """加载所有必需的模型"""
        self.logger.info("开始加载模型...")
        
        try:
            # 加载CLIP和Stable Diffusion模型
            self.clip_model, self.clip_processor, self.sd_pipeline = self.model_loader.load_all_models()
            
            # 初始化各个模块
            self.semantic_hash_generator = SemanticHashGenerator(
                self.clip_model, self.clip_processor
            )
            
            self.base_watermark_generator = BaseWatermarkGenerator(self.device)
            
            self.attention_extractor = AttentionExtractor(self.device)
            
            self.message_packet_generator = MessagePacketGenerator()
            
            self.logger.info("所有模型和模块加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def demonstrate_task1(self, prompt: str) -> str:
        """
        演示任务1：语义哈希生成
        
        Args:
            prompt: 输入prompt
            
        Returns:
            生成的语义哈希
        """
        self.logger.info("=" * 40)
        self.logger.info("任务1：语义哈希生成演示")
        self.logger.info("=" * 40)
        
        # 生成语义哈希
        c_bits = self.semantic_hash_generator.generate_semantic_hash(prompt)
        
        # 验证一致性
        self.logger.info("验证哈希一致性...")
        is_consistent = self.semantic_hash_generator.verify_hash_consistency(prompt, num_tests=3)
        
        # 比较不同prompt
        test_prompt = "a cat sitting on a chair"
        self.logger.info(f"与测试prompt比较: '{test_prompt}'")
        hash1, hash2, hamming_dist = self.semantic_hash_generator.compare_prompts(prompt, test_prompt)
        
        self.logger.info(f"任务1完成 - 语义哈希: {c_bits[:32]}...{c_bits[-32:]}")
        return c_bits
    
    def demonstrate_task2(self, c_bits: str) -> torch.Tensor:
        """
        演示任务2：基准水印生成
        
        Args:
            c_bits: 语义哈希
            
        Returns:
            生成的基准水印
        """
        self.logger.info("=" * 40)
        self.logger.info("任务2：基准水印生成演示")
        self.logger.info("=" * 40)
        
        # 生成基准水印
        w_base = self.base_watermark_generator.generate_base_watermark(c_bits)
        
        # 验证确定性
        self.logger.info("验证确定性生成...")
        is_deterministic = self.base_watermark_generator.verify_deterministic_generation(
            c_bits, num_tests=3
        )
        
        # 获取统计信息
        stats = self.base_watermark_generator.get_watermark_statistics(w_base)
        self.logger.info(f"水印统计: 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}")
        
        self.logger.info(f"任务2完成 - 基准水印形状: {w_base.shape}")
        return w_base
    
    def demonstrate_task3(self, prompt: str) -> tuple:
        """
        演示任务3：注意力图谱提取与掩码生成
        
        Args:
            prompt: 输入prompt
            
        Returns:
            (注意力图谱, 语义掩码, 生成的图像)
        """
        self.logger.info("=" * 40)
        self.logger.info("任务3：注意力图谱提取与掩码生成演示")
        self.logger.info("=" * 40)
        
        # 提取注意力并生成掩码
        attention_map, semantic_mask, generated_image = self.attention_extractor.extract_and_generate_mask(
            self.sd_pipeline, prompt, num_inference_steps=10  # 减少步数以加快演示
        )
        
        # 保存可视化结果
        ensure_dir("outputs")
        vis_path = "outputs/attention_visualization.png"
        self.attention_extractor.visualize_attention_and_mask(
            attention_map, semantic_mask, vis_path
        )
        
        # 保存生成的图像
        generated_image.save("outputs/generated_image.png")
        
        self.logger.info(f"任务3完成 - 注意力图谱形状: {attention_map.shape}, "
                        f"语义掩码形状: {semantic_mask.shape}")
        return attention_map, semantic_mask, generated_image
    
    def demonstrate_task4(self, c_bits: str, copyright_info: str) -> str:
        """
        演示任务4：信息包生成
        
        Args:
            c_bits: 语义哈希
            copyright_info: 版权信息
            
        Returns:
            生成的信息包
        """
        self.logger.info("=" * 40)
        self.logger.info("任务4：信息包生成演示")
        self.logger.info("=" * 40)
        
        # 生成信息包
        message_packet = self.message_packet_generator.create_message_packet(c_bits, copyright_info)
        
        # 验证完整性
        self.logger.info("验证信息包完整性...")
        is_valid = self.message_packet_generator.verify_packet_integrity(c_bits, copyright_info)
        
        # 获取配置信息
        packet_info = self.message_packet_generator.get_packet_info()
        self.logger.info(f"BCH参数: n={packet_info['bch_n']}, k={packet_info['bch_k']}, t={packet_info['bch_t']}")
        
        self.logger.info(f"任务4完成 - 信息包长度: {len(message_packet)}位")
        return message_packet
    
    def run_complete_demo(self, prompt: str = "一只戴着宇航头盔的柴犬", copyright_info: str = "UserID:12345"):
        """
        运行完整的演示流程
        
        Args:
            prompt: 演示用的prompt
            copyright_info: 演示用的版权信息
        """
        self.logger.info("开始完整演示流程...")
        self.logger.info(f"演示prompt: '{prompt}'")
        self.logger.info(f"版权信息: '{copyright_info}'")
        
        try:
            # 加载模型
            self.load_models()
            
            # 任务1：语义哈希生成
            c_bits = self.demonstrate_task1(prompt)
            
            # 任务2：基准水印生成
            w_base = self.demonstrate_task2(c_bits)
            
            # 任务3：注意力图谱提取与掩码生成
            attention_map, semantic_mask, generated_image = self.demonstrate_task3(prompt)
            
            # 任务4：信息包生成
            message_packet = self.demonstrate_task4(c_bits, copyright_info)
            
            # 总结
            self.logger.info("=" * 60)
            self.logger.info("完整演示流程执行成功！")
            self.logger.info("=" * 60)
            self.logger.info("生成的核心信息:")
            self.logger.info(f"1. 语义哈希 (c_bits): {len(c_bits)}位")
            self.logger.info(f"2. 基准水印 (W_base): {w_base.shape}")
            self.logger.info(f"3. 语义掩码 (M_sem_binary): {semantic_mask.shape}")
            self.logger.info(f"4. 信息包 (M): {len(message_packet)}位")
            self.logger.info("输出文件:")
            self.logger.info("- outputs/generated_image.png (生成的图像)")
            self.logger.info("- outputs/attention_visualization.png (注意力可视化)")
            
        except Exception as e:
            self.logger.error(f"演示流程执行失败: {str(e)}")
            raise
        finally:
            # 清理模型释放内存
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("正在清理资源...")
        if self.model_loader:
            self.model_loader.clear_models()
        self.logger.info("资源清理完成")


def main():
    """主函数"""
    # 确保输出目录存在
    ensure_dir("outputs")
    ensure_dir("logs")
    
    # 创建并运行演示
    demo = SAHWMBaseline()
    
    try:
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
