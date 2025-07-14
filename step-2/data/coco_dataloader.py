"""
SAAH-WM Baseline 第二步 - COCO数据集加载器

本模块实现COCO2017数据集的加载，并集成第一步的四个核心模块：
- 语义哈希生成器
- 基准水印生成器  
- 信息包生成器
- 注意力提取器（训练时不使用，为第三步准备）

每个训练样本包含：
- 原始图像
- 动态生成的语义哈希c_bits
- 基于c_bits的基准水印W_base
- 包含版权信息的信息包M
"""

import os
import random
import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 导入第一步的核心模块
import sys
sys.path.append('../step-1')
try:
    from core.semantic_hash import SemanticHashGenerator
    from core.base_watermark import BaseWatermarkGenerator
    from core.message_packet import MessagePacketGenerator
    from utils.model_loader import ModelLoader
except ImportError as e:
    logging.warning(f"无法导入第一步模块: {e}")
    logging.warning("将使用模拟数据代替")

# 配置日志
logger = logging.getLogger(__name__)


class COCOWatermarkDataset(data.Dataset):
    """
    COCO水印数据集
    
    集成第一步的核心模块，为每个图像动态生成训练所需的水印信息。
    
    Args:
        coco_path (str): COCO2017数据集路径
        image_size (int): 图像尺寸，默认512
        crop_size (int): 随机裁剪尺寸，默认400
        split (str): 数据集分割，'train'或'val'，默认'train'
        use_step1_modules (bool): 是否使用第一步模块，默认True
        latent_size (int): 潜在空间尺寸，默认64（对应512图像）
    """
    
    def __init__(self, 
                 coco_path: str,
                 image_size: int = 512,
                 crop_size: int = 400,
                 split: str = 'train',
                 use_step1_modules: bool = True,
                 latent_size: int = 64):
        
        super(COCOWatermarkDataset, self).__init__()
        
        self.coco_path = coco_path
        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.use_step1_modules = use_step1_modules
        self.latent_size = latent_size
        
        logger.info(f"初始化COCO水印数据集: 路径={coco_path}, 图像尺寸={image_size}, "
                   f"裁剪尺寸={crop_size}, 分割={split}")
        
        # 获取图像文件列表
        self.image_files = self._get_image_files()
        logger.info(f"找到 {len(self.image_files)} 张图像")
        
        # 初始化图像变换
        self._setup_transforms()
        
        # 初始化第一步模块
        if self.use_step1_modules:
            self._setup_step1_modules()
        else:
            logger.warning("未使用第一步模块，将生成模拟数据")
            
        # 版权信息模板
        self.copyright_templates = [
            "UserID:12345",
            "Copyright:SAAH-WM",
            "Owner:Research",
            "License:MIT",
            "Project:Baseline"
        ]
        
    def _get_image_files(self) -> List[str]:
        """获取图像文件列表"""
        if not os.path.exists(self.coco_path):
            raise FileNotFoundError(f"COCO数据集路径不存在: {self.coco_path}")
            
        # 支持的图像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        image_files = []
        for filename in os.listdir(self.coco_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(filename)
                
        if not image_files:
            raise ValueError(f"在路径 {self.coco_path} 中未找到有效的图像文件")
            
        # 随机打乱
        random.shuffle(image_files)
        
        return image_files
        
    def _setup_transforms(self):
        """设置图像变换"""
        if self.split == 'train':
            # 训练时的数据增强
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
            ])
        else:
            # 验证时不使用数据增强
            self.transform = transforms.Compose([
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
            ])
            
        # 潜在空间变换（模拟Stable Diffusion的VAE编码）
        self.latent_transform = transforms.Compose([
            transforms.Resize((self.latent_size, self.latent_size)),
            transforms.Normalize(mean=[0.0, 0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0, 1.0])
        ])
        
    def _setup_step1_modules(self):
        """初始化第一步模块"""
        try:
            # 初始化模型加载器
            self.model_loader = ModelLoader()
            
            # 初始化语义哈希生成器
            self.semantic_hash_generator = SemanticHashGenerator(
                clip_model_name="openai/clip-vit-large-patch14",
                hash_length=256,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 初始化基准水印生成器
            self.base_watermark_generator = BaseWatermarkGenerator()
            
            # 初始化信息包生成器
            self.message_packet_generator = MessagePacketGenerator(
                bch_polynomial=285,
                bch_bits=8
            )
            
            logger.info("第一步模块初始化完成")
            
        except Exception as e:
            logger.error(f"第一步模块初始化失败: {e}")
            self.use_step1_modules = False
            
    def _generate_random_prompt(self) -> str:
        """生成随机prompt用于语义哈希"""
        subjects = ["a dog", "a cat", "a bird", "a car", "a house", "a tree", "a flower", "a person"]
        adjectives = ["beautiful", "colorful", "amazing", "wonderful", "fantastic", "incredible"]
        styles = ["realistic", "artistic", "detailed", "high quality", "professional"]
        
        subject = random.choice(subjects)
        adjective = random.choice(adjectives)
        style = random.choice(styles)
        
        return f"{adjective} {subject}, {style}"
        
    def _generate_copyright_info(self) -> str:
        """生成随机版权信息"""
        template = random.choice(self.copyright_templates)
        
        # 添加随机数字使每个样本的版权信息唯一
        random_id = random.randint(10000, 99999)
        return f"{template}:{random_id}"
        
    def _simulate_latent_encoding(self, image: torch.Tensor) -> torch.Tensor:
        """
        模拟Stable Diffusion VAE编码过程
        
        Args:
            image (torch.Tensor): RGB图像，形状 [3, H, W]
            
        Returns:
            torch.Tensor: 潜在特征，形状 [4, latent_size, latent_size]
        """
        # 简化的潜在空间编码模拟
        # 实际应用中应该使用真正的Stable Diffusion VAE
        
        # 调整到潜在空间尺寸
        resized = transforms.Resize((self.latent_size, self.latent_size))(image)
        
        # 扩展到4通道（Stable Diffusion潜在空间）
        # 使用简单的线性组合模拟
        r, g, b = resized[0], resized[1], resized[2]
        
        latent = torch.stack([
            r * 0.5 + g * 0.3 + b * 0.2,  # 通道1
            r * 0.2 + g * 0.5 + b * 0.3,  # 通道2  
            r * 0.3 + g * 0.2 + b * 0.5,  # 通道3
            (r + g + b) / 3.0              # 通道4
        ], dim=0)
        
        return latent
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含所有训练数据的字典
        """
        try:
            # 加载图像
            image_path = os.path.join(self.coco_path, self.image_files[index])
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            image_tensor = self.transform(image)  # [3, crop_size, crop_size]
            
            # 转换到潜在空间
            image_latent = self._simulate_latent_encoding(image_tensor)  # [4, latent_size, latent_size]
            
            if self.use_step1_modules:
                # 使用第一步模块生成数据
                
                # 生成随机prompt和语义哈希
                prompt = self._generate_random_prompt()
                c_bits = self.semantic_hash_generator.generate_semantic_hash(prompt)
                
                # 生成基准水印
                base_watermark = self.base_watermark_generator.generate_base_watermark(
                    c_bits, self.latent_size, self.latent_size, 4
                )  # [1, 4, latent_size, latent_size]
                base_watermark = base_watermark.squeeze(0)  # [4, latent_size, latent_size]
                
                # 生成版权信息和信息包
                copyright_info = self._generate_copyright_info()
                message_packet = self.message_packet_generator.create_message_packet(c_bits, copyright_info)
                
                # 转换信息包为张量
                message_bits = torch.tensor([float(bit) for bit in message_packet], dtype=torch.float32)
                
            else:
                # 生成模拟数据
                c_bits = ''.join([str(random.randint(0, 1)) for _ in range(256)])
                base_watermark = torch.randn(4, self.latent_size, self.latent_size)
                message_bits = torch.randint(0, 2, (32,)).float()  # 32位信息包
                
            # 构建返回数据
            sample = {
                'image_latent': image_latent,           # 原始图像潜在特征 [4, H, W]
                'base_watermark': base_watermark,       # 基准水印 [4, H, W]
                'message_bits': message_bits,           # 信息包 [message_dim]
                'c_bits': c_bits,                       # 语义哈希字符串
                'image_path': image_path                # 图像路径（用于调试）
            }
            
            logger.debug(f"样本 {index} 加载完成: {self.image_files[index]}")
            
            return sample
            
        except Exception as e:
            logger.error(f"加载样本 {index} 时发生错误: {e}")
            # 返回随机数据作为备用
            return self._get_fallback_sample()
            
    def _get_fallback_sample(self) -> Dict[str, torch.Tensor]:
        """获取备用样本（随机数据）"""
        return {
            'image_latent': torch.randn(4, self.latent_size, self.latent_size),
            'base_watermark': torch.randn(4, self.latent_size, self.latent_size),
            'message_bits': torch.randint(0, 2, (32,)).float(),
            'c_bits': ''.join([str(random.randint(0, 1)) for _ in range(256)]),
            'image_path': 'fallback_sample'
        }
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)
        
    def get_dataset_info(self) -> Dict[str, any]:
        """获取数据集信息"""
        return {
            'dataset_size': len(self.image_files),
            'coco_path': self.coco_path,
            'image_size': self.image_size,
            'crop_size': self.crop_size,
            'latent_size': self.latent_size,
            'split': self.split,
            'use_step1_modules': self.use_step1_modules
        }


def create_dataloader(coco_path: str,
                     batch_size: int = 4,
                     num_workers: int = 8,
                     pin_memory: bool = True,
                     shuffle: bool = True,
                     split: str = 'train',
                     **dataset_kwargs) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        coco_path (str): COCO数据集路径
        batch_size (int): 批次大小，默认4
        num_workers (int): 工作进程数，默认8
        pin_memory (bool): 是否固定内存，默认True
        shuffle (bool): 是否打乱数据，默认True
        split (str): 数据集分割，默认'train'
        **dataset_kwargs: 传递给数据集的其他参数
        
    Returns:
        DataLoader: 配置好的数据加载器
    """
    
    # 创建数据集
    dataset = COCOWatermarkDataset(
        coco_path=coco_path,
        split=split,
        **dataset_kwargs
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 确保批次大小一致
    )
    
    logger.info(f"数据加载器创建完成: 批次大小={batch_size}, 工作进程={num_workers}, "
               f"数据集大小={len(dataset)}, 批次数={len(dataloader)}")
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试配置
    test_config = {
        'coco_path': "D:/CodePaper/EditGuard/train2017/train2017",  # 请根据实际路径修改
        'batch_size': 2,
        'num_workers': 0,  # 测试时使用0避免多进程问题
        'image_size': 512,
        'crop_size': 400,
        'latent_size': 64,
        'use_step1_modules': False  # 测试时使用模拟数据
    }
    
    try:
        # 创建数据加载器
        dataloader = create_dataloader(**test_config)
        
        # 获取数据集信息
        dataset_info = dataloader.dataset.get_dataset_info()
        print("数据集信息:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")
        
        # 测试数据加载
        print(f"\n开始测试数据加载...")
        for i, batch in enumerate(dataloader):
            print(f"\n批次 {i+1}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} | 类型: {value.dtype} | "
                          f"范围: [{value.min().item():.4f}, {value.max().item():.4f}]")
                elif isinstance(value, list):
                    print(f"  {key}: 列表长度={len(value)} | 示例: {value[0][:50]}...")
                else:
                    print(f"  {key}: {type(value)} | 值: {value}")
            
            # 只测试前2个批次
            if i >= 1:
                break
                
        print("\n数据加载器测试完成!")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请检查COCO数据集路径是否正确")
        
        # 使用模拟数据进行基本测试
        print("\n使用模拟数据进行基本测试...")
        
        # 创建临时测试目录和文件
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一些测试图像文件
            test_image = Image.new('RGB', (256, 256), color='red')
            for i in range(5):
                test_image.save(os.path.join(temp_dir, f'test_{i}.jpg'))
            
            # 使用临时目录测试
            test_config['coco_path'] = temp_dir
            dataloader = create_dataloader(**test_config)
            
            # 测试一个批次
            batch = next(iter(dataloader))
            print("模拟数据测试成功!")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                    
        print("基本测试完成!")
