"""
语义哈希生成器测试

测试语义哈希生成的一致性、唯一性和正确性。

作者：SAAH-WM团队
"""

import unittest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import ModelLoader
from core.semantic_hash import SemanticHashGenerator


class TestSemanticHashGenerator(unittest.TestCase):
    """语义哈希生成器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        print("正在加载模型进行测试...")
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model_loader = ModelLoader(cls.device)
        
        try:
            cls.clip_model, cls.clip_processor, _ = cls.model_loader.load_all_models()
            cls.hash_generator = SemanticHashGenerator(cls.clip_model, cls.clip_processor)
            print("模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        if hasattr(cls, 'model_loader'):
            cls.model_loader.clear_models()
        print("测试完成，模型已清理")
    
    def test_hash_generation(self):
        """测试基本哈希生成功能"""
        prompt = "a dog sitting in the park"
        hash_result = self.hash_generator.generate_semantic_hash(prompt)
        
        # 检查哈希长度
        self.assertEqual(len(hash_result), 256, "哈希长度应为256位")
        
        # 检查是否为二进制字符串
        self.assertTrue(all(c in '01' for c in hash_result), "哈希应为二进制字符串")
        
        print(f"✓ 基本哈希生成测试通过，哈希: {hash_result[:32]}...")
    
    def test_hash_consistency(self):
        """测试哈希一致性"""
        prompt = "a cat playing with a ball"
        
        # 生成多次哈希
        hashes = []
        for i in range(5):
            hash_result = self.hash_generator.generate_semantic_hash(prompt)
            hashes.append(hash_result)
        
        # 检查所有哈希是否相同
        reference_hash = hashes[0]
        for i, hash_result in enumerate(hashes[1:], 1):
            self.assertEqual(hash_result, reference_hash, 
                           f"第{i+1}次生成的哈希与参考哈希不同")
        
        print("✓ 哈希一致性测试通过")
    
    def test_hash_uniqueness(self):
        """测试不同prompt生成不同哈希"""
        prompts = [
            "a dog running in the field",
            "a cat sleeping on the sofa", 
            "a bird flying in the sky",
            "a fish swimming in the ocean"
        ]
        
        hashes = []
        for prompt in prompts:
            hash_result = self.hash_generator.generate_semantic_hash(prompt)
            hashes.append(hash_result)
        
        # 检查所有哈希都不相同
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                self.assertNotEqual(hashes[i], hashes[j], 
                                  f"Prompt '{prompts[i]}' 和 '{prompts[j]}' 生成了相同的哈希")
        
        print("✓ 哈希唯一性测试通过")
    
    def test_similar_prompts(self):
        """测试相似prompt的哈希差异"""
        prompt1 = "a red car on the road"
        prompt2 = "a blue car on the road"
        
        hash1 = self.hash_generator.generate_semantic_hash(prompt1)
        hash2 = self.hash_generator.generate_semantic_hash(prompt2)
        
        # 计算汉明距离
        hamming_distance = sum(b1 != b2 for b1, b2 in zip(hash1, hash2))
        
        # 相似prompt应该有一定差异但不会完全不同
        self.assertGreater(hamming_distance, 0, "相似prompt应该产生不同的哈希")
        self.assertLess(hamming_distance, 256, "相似prompt的哈希不应该完全不同")
        
        print(f"✓ 相似prompt测试通过，汉明距离: {hamming_distance}/256")
    
    def test_empty_and_special_prompts(self):
        """测试空prompt和特殊字符prompt"""
        special_prompts = [
            "",  # 空prompt
            " ",  # 空格
            "a",  # 单字符
            "123",  # 数字
            "!@#$%",  # 特殊字符
        ]
        
        for prompt in special_prompts:
            try:
                hash_result = self.hash_generator.generate_semantic_hash(prompt)
                self.assertEqual(len(hash_result), 256, f"特殊prompt '{prompt}' 哈希长度错误")
                print(f"✓ 特殊prompt '{prompt}' 测试通过")
            except Exception as e:
                self.fail(f"特殊prompt '{prompt}' 处理失败: {e}")
    
    def test_hash_info(self):
        """测试哈希生成器信息获取"""
        info = self.hash_generator.get_hash_info()
        
        # 检查必要的信息字段
        required_fields = ['hash_bits', 'embedding_dim', 'random_seed', 'device']
        for field in required_fields:
            self.assertIn(field, info, f"缺少信息字段: {field}")
        
        # 检查数值的合理性
        self.assertEqual(info['hash_bits'], 256, "哈希位数应为256")
        self.assertGreater(info['embedding_dim'], 0, "嵌入维度应大于0")
        
        print(f"✓ 哈希信息测试通过: {info}")


def run_semantic_hash_tests():
    """运行语义哈希测试"""
    print("=" * 50)
    print("开始语义哈希生成器测试")
    print("=" * 50)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSemanticHashGenerator)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n✅ 所有语义哈希测试通过！")
    else:
        print(f"\n❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_semantic_hash_tests()
