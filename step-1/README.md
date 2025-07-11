# SAAH-WM Baseline 第一步实现

## 项目概述

本项目实现了SAAH-WM（语义感知与自适应分层水印）基线版本的第一步任务，包含四个核心模块的完整实现：

1. **语义哈希生成** - 基于CLIP模型和SimHash算法
2. **基准水印生成** - 使用确定性伪随机数生成器
3. **注意力图谱提取与掩码生成** - 从Stable Diffusion U-Net提取注意力并生成二值掩码
4. **信息包生成** - 使用BCH纠错码组合语义哈希和版权信息

## 技术特性

- ✅ **模块化设计** - 四个独立模块，便于测试和扩展
- ✅ **详细日志** - 完整的日志记录，包括模型加载、处理进度、错误处理
- ✅ **中文注释** - 所有代码都有详细的中文注释和文档
- ✅ **完整测试** - 每个模块都有对应的单元测试
- ✅ **错误处理** - 完善的异常捕获和友好的错误提示
- ✅ **内存优化** - 支持GPU内存优化和模型清理

## 环境要求

### 硬件要求
- **推荐**: NVIDIA GPU (8GB+ VRAM)
- **最低**: CPU (运行较慢)

### 软件要求
- Python 3.8+
- CUDA 11.0+ (如果使用GPU)

## 安装指南

### 1. 克隆项目
```bash
cd step-1
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 快速开始

### 运行完整演示
```bash
python main.py
```

这将执行完整的四步流程：
1. 加载Stable Diffusion 2.1和CLIP模型
2. 生成语义哈希
3. 生成基准水印
4. 提取注意力图谱并生成掩码
5. 创建信息包

### 运行测试
```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行单个模块测试
python tests/test_semantic_hash.py
python tests/test_base_watermark.py
python tests/test_message_packet.py
```

## 模块详解

### 1. 语义哈希生成器 (`core/semantic_hash.py`)

**功能**: 将文本prompt转换为256位二进制哈希

**核心特性**:
- 使用CLIP模型编码文本为768维语义向量
- 复用SEAL项目的SimHash算法
- 确保相同prompt生成相同哈希
- 不同prompt生成不同哈希

**使用示例**:
```python
from core.semantic_hash import SemanticHashGenerator

# 初始化（需要预加载的CLIP模型）
generator = SemanticHashGenerator(clip_model, clip_processor)

# 生成语义哈希
c_bits = generator.generate_semantic_hash("一只戴着宇航头盔的柴犬")
print(f"语义哈希: {c_bits}")  # 256位二进制字符串
```

### 2. 基准水印生成器 (`core/base_watermark.py`)

**功能**: 使用语义哈希作为种子生成确定性水印

**核心特性**:
- 确定性生成：相同输入产生相同水印
- 支持自定义尺寸（默认64x64x4）
- 高频随机水印，适合脆弱性检测

**使用示例**:
```python
from core.base_watermark import BaseWatermarkGenerator

generator = BaseWatermarkGenerator(device="cuda")
w_base = generator.generate_base_watermark(c_bits, 64, 64, 4)
print(f"基准水印形状: {w_base.shape}")  # [1, 4, 64, 64]
```

### 3. 注意力图谱提取器 (`core/attention_extractor.py`)

**功能**: 从Stable Diffusion提取注意力图谱并生成语义掩码

**核心特性**:
- 非侵入式钩子机制提取交叉注意力
- 聚合多层多头注意力图谱
- 使用Otsu方法生成二值掩码
- 支持可视化和保存

**使用示例**:
```python
from core.attention_extractor import AttentionExtractor

extractor = AttentionExtractor(device="cuda")
attention_map, mask, image = extractor.extract_and_generate_mask(
    sd_pipeline, "一只戴着宇航头盔的柴犬"
)
print(f"语义掩码形状: {mask.shape}")  # [1, 1, 64, 64]
```

### 4. 信息包生成器 (`core/message_packet.py`)

**功能**: 组合语义哈希和版权信息，添加BCH纠错码

**核心特性**:
- 支持最大64字符版权信息
- BCH纠错码保证数据完整性
- 支持编码和解码验证
- 自动处理字符编码和填充

**使用示例**:
```python
from core.message_packet import MessagePacketGenerator

generator = MessagePacketGenerator()
packet = generator.create_message_packet(c_bits, "UserID:12345")
print(f"信息包长度: {len(packet)}位")

# 验证完整性
is_valid = generator.verify_packet_integrity(c_bits, "UserID:12345")
print(f"完整性验证: {is_valid}")
```

## 输出文件

运行演示后，会在`outputs/`目录生成：

- `generated_image.png` - Stable Diffusion生成的图像
- `attention_visualization.png` - 注意力图谱和掩码可视化
- `logs/saah_wm_*.log` - 详细的运行日志

## 配置选项

### 模型配置
```python
# 在main.py中修改
CLIP_MODEL = "openai/clip-vit-large-patch14"
SD_MODEL = "stabilityai/stable-diffusion-2-1-base"
```

### 哈希配置
```python
# 在semantic_hash.py中修改
HASH_BITS = 256  # 哈希位数
RANDOM_SEED = 42  # 超平面生成种子
```

### BCH配置
```python
# 在message_packet.py中修改
BCH_POLYNOMIAL = 137  # BCH多项式
BCH_BITS = 5  # 纠错位数
```

## 性能优化

### GPU内存优化
- 自动启用attention slicing
- 支持model CPU offload
- 及时清理模型释放内存

### 推理加速
```python
# 减少推理步数（在演示中）
num_inference_steps = 10  # 默认20

# 使用较小的引导尺度
guidance_scale = 5.0  # 默认7.5
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：使用CPU或减少批次大小
   export CUDA_VISIBLE_DEVICES=""  # 强制使用CPU
   ```

2. **模型下载失败**
   ```bash
   # 解决方案：设置代理或使用镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **BCH库安装失败**
   ```bash
   # 解决方案：安装编译工具
   pip install --upgrade setuptools wheel
   pip install bchlib
   ```

### 调试模式
```python
# 在utils/logger_config.py中修改日志级别
setup_logger(log_level="DEBUG")
```

## 扩展开发

### 添加新的哈希算法
1. 在`core/semantic_hash.py`中添加新方法
2. 在`tests/test_semantic_hash.py`中添加测试
3. 更新文档

### 集成新的水印方法
1. 参考`core/base_watermark.py`的接口设计
2. 实现相应的测试用例
3. 在主程序中集成

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: [GitHub Issues]
- 邮箱: [项目邮箱]

---

**SAAH-WM团队**  
版本: 1.0.0  
更新日期: 2024年
