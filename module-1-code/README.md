# SAAH-WM 模块一：语义VQ-VAE

这是SAAH-WM项目的第一个模块，实现了语义指纹生成与压缩功能。该模块将768维的CLIP语义向量压缩为离散的二进制索引，支持精确重构。

## 功能特性

- **高质量压缩**：将768维CLIP向量压缩为10-bit索引（1024个码本向量）
- **精确重构**：使用VQ-VAE确保相同索引总是重构出完全相同的向量
- **EMA更新**：采用指数移动平均更新码本，训练更稳定
- **完整流水线**：从数据预处理到模型训练再到测试评估的完整实现

## 项目结构

```
module-1-code/
├── models/                 # 模型定义
│   ├── vqvae_model.py     # 主要VQ-VAE模型
│   └── quantizer.py       # EMA矢量量化器
├── data/                  # 数据处理
│   ├── dataset.py         # 数据集类
│   └── preprocess_data.py # 数据预处理脚本
├── training/              # 训练相关
│   ├── train.py          # 训练脚本
│   └── config.py         # 配置管理
├── evaluation/            # 测试评估
│   ├── test.py           # 测试脚本
│   └── metrics.py        # 评估指标
├── utils/                 # 工具函数
│   └── clip_utils.py     # CLIP相关工具
└── requirements.txt       # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
cd module-1-code
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 下载并预处理conceptual_captions数据集
python data/preprocess_data.py --max_samples 50000 --output_dir ./data/processed
```

### 3. 训练模型

```bash
# 使用默认配置训练
python training/train.py --config default --data_dir ./data/processed --output_dir ./outputs

# 使用小规模配置（快速测试）
python training/train.py --config small --data_dir ./data/processed --output_dir ./outputs

# 使用大规模配置（更好性能）
python training/train.py --config large --data_dir ./data/processed --output_dir ./outputs
```

### 4. 测试模型

```bash
# 测试训练好的模型
python evaluation/test.py --model_path ./outputs/semantic_vqvae/best_checkpoint.pt --output_dir ./test_results
```

## 配置说明

### 模型配置

- `input_dim`: 输入维度（默认768，CLIP向量维度）
- `latent_dim`: 潜在空间维度（默认256）
- `num_embeddings`: 码本大小（默认1024）
- `commitment_cost`: 承诺损失权重（默认0.25）
- `decay`: EMA衰减系数（默认0.99）

### 训练配置

- `batch_size`: 批处理大小（默认128）
- `learning_rate`: 学习率（默认1e-3）
- `num_epochs`: 训练轮数（默认100）
- `mixed_precision`: 混合精度训练（默认True）

## 核心API

### 模型使用

```python
from models.vqvae_model import SemanticVQVAE
from utils.clip_utils import CLIPTextEncoder

# 创建模型
model = SemanticVQVAE(
    input_dim=768,
    latent_dim=256,
    num_embeddings=1024
)

# 创建CLIP编码器
clip_encoder = CLIPTextEncoder()

# 文本到比特串
prompt = "a cat sitting on a chair"
clip_vector = clip_encoder.encode_text(prompt)
indices = model.encode(clip_vector)
bits = format(indices[0].item(), '010b')

# 比特串到重构向量
index = int(bits, 2)
reconstructed = model.decode_from_indices(torch.tensor([index]))
```

### 完整流水线

```python
from utils.clip_utils import prompt_to_bits, bits_to_reconstructed_vector

# 压缩
bits = prompt_to_bits("a beautiful sunset", model, clip_encoder)

# 重构
reconstructed_vector = bits_to_reconstructed_vector(bits, model)
```

## 评估指标

模型评估包括以下指标：

- **重构质量**：余弦相似度、MSE损失、L2距离
- **码本利用率**：使用的码本向量比例
- **压缩效率**：压缩比、存储节省
- **语义保持性**：聚类一致性、轮廓系数
- **鲁棒性**：噪声稳定性、编码稳定性

## 预期性能

训练良好的模型应该达到：

- 余弦相似度 > 0.95
- 码本利用率 > 80%
- 压缩比：768×32 / 10 = 2457.6:1
- 重构一致性：100%（相同索引总是产生相同向量）

## 故障排除

### 常见问题

1. **CUDA内存不足**：减小batch_size或使用CPU训练
2. **数据集下载失败**：检查网络连接，或使用更小的数据集
3. **码本利用率低**：增加训练轮数或调整学习率
4. **重构质量差**：增加码本大小或潜在维度

### 调试技巧

- 使用`config="small"`进行快速测试
- 检查数据预处理是否正确完成
- 监控训练日志中的损失变化
- 使用TensorBoard查看训练曲线

## 下一步

完成模块一后，可以继续开发：

- 模块二：语义重要性分析（交叉注意力图谱提取）
- 模块三：自适应分层水印嵌入
- 模块四：智能篡改定位

## 参考文献

- VQ-VAE: Neural Discrete Representation Learning
- CLIP: Learning Transferable Visual Representations
- SAAH-WM项目总览文档
