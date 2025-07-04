"""
数据预处理脚本
从conceptual_captions数据集提取CLIP语义向量
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clip_utils import CLIPTextEncoder


def preprocess_conceptual_captions(
    output_dir="./data/processed",
    max_samples=100000,
    batch_size=64,
    device="cuda"
):
    """
    预处理conceptual_captions数据集
    Args:
        output_dir: 输出目录
        max_samples: 最大样本数
        batch_size: 批处理大小
        device: 计算设备
    """
    print("Loading conceptual_captions dataset...")
    
    # 加载数据集
    try:
        dataset = load_dataset("conceptual_captions", split="train")
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load a smaller subset...")
        # 如果加载失败，尝试加载更小的数据集
        dataset = load_dataset("conceptual_captions", split="train[:10000]")
    
    # 限制样本数量
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")
    
    # 初始化CLIP编码器
    print("Initializing CLIP encoder...")
    clip_encoder = CLIPTextEncoder(device=device)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取文本
    texts = [item['caption'] for item in dataset]
    print(f"Extracted {len(texts)} captions")
    
    # 批量编码
    print("Encoding texts to semantic vectors...")
    all_vectors = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # 编码当前批次
            batch_vectors = clip_encoder.encode_text(batch_texts, normalize=True)
            all_vectors.append(batch_vectors.cpu())
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
    
    # 合并所有向量
    if all_vectors:
        semantic_vectors = torch.cat(all_vectors, dim=0)
        print(f"Generated {semantic_vectors.shape[0]} semantic vectors of dimension {semantic_vectors.shape[1]}")
        
        # 保存向量
        output_path = os.path.join(output_dir, "clip_vectors.pt")
        torch.save(semantic_vectors, output_path)
        print(f"Saved semantic vectors to {output_path}")
        
        # 保存对应的文本（用于调试）
        texts_path = os.path.join(output_dir, "texts.txt")
        with open(texts_path, 'w', encoding='utf-8') as f:
            for text in texts[:semantic_vectors.shape[0]]:
                f.write(text + '\n')
        print(f"Saved corresponding texts to {texts_path}")
        
        # 保存统计信息
        stats = {
            'num_samples': semantic_vectors.shape[0],
            'vector_dim': semantic_vectors.shape[1],
            'mean': semantic_vectors.mean(dim=0),
            'std': semantic_vectors.std(dim=0),
            'min': semantic_vectors.min(dim=0)[0],
            'max': semantic_vectors.max(dim=0)[0]
        }
        
        stats_path = os.path.join(output_dir, "statistics.pt")
        torch.save(stats, stats_path)
        print(f"Saved statistics to {stats_path}")
        
        return semantic_vectors
    else:
        print("No vectors were generated!")
        return None


def create_train_val_split(
    vectors_path="./data/processed/clip_vectors.pt",
    output_dir="./data/processed",
    val_ratio=0.1,
    random_seed=42
):
    """
    创建训练/验证集划分
    Args:
        vectors_path: 向量文件路径
        output_dir: 输出目录
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    print("Creating train/validation split...")
    
    # 加载向量
    vectors = torch.load(vectors_path)
    print(f"Loaded {vectors.shape[0]} vectors")
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    
    # 随机打乱
    indices = torch.randperm(vectors.shape[0])
    vectors = vectors[indices]
    
    # 划分
    val_size = int(vectors.shape[0] * val_ratio)
    train_vectors = vectors[val_size:]
    val_vectors = vectors[:val_size]
    
    # 保存
    train_path = os.path.join(output_dir, "train_vectors.pt")
    val_path = os.path.join(output_dir, "val_vectors.pt")
    
    torch.save(train_vectors, train_path)
    torch.save(val_vectors, val_path)
    
    print(f"Saved {train_vectors.shape[0]} training vectors to {train_path}")
    print(f"Saved {val_vectors.shape[0]} validation vectors to {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess conceptual_captions dataset")
    parser.add_argument("--output_dir", default="./data/processed", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # 预处理数据
    vectors = preprocess_conceptual_captions(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if vectors is not None:
        # 创建训练/验证集划分
        vectors_path = os.path.join(args.output_dir, "clip_vectors.pt")
        create_train_val_split(
            vectors_path=vectors_path,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio
        )
        
        print("Data preprocessing completed successfully!")
    else:
        print("Data preprocessing failed!")


if __name__ == "__main__":
    main()
