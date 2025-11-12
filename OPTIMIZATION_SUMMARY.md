# LED模型优化总结

本文档总结了为"What's Up, Docs?"论文摘要比赛所做的所有优化。

## 文件改动总结

### 1. 修改的文件

#### `configs/led_base_config.yaml`
- **模型升级**: 将`model_name`从`allenai/led-base-16384`改为`allenai/led-large-16384`
- **批次大小**: `batch_size`保持为1（大模型内存限制）
- **梯度累积**: `gradient_accumulation_steps`从4调整为16（有效批次大小=16）
- **学习率**: `learning_rate`从3.0e-5调整为2.0e-5（大模型需要更小的学习率）
- **训练轮数**: `epochs`从1调整为5

#### `src/train_led_large.py`
- **新增函数**:
  - `create_academic_prompt_structured()`: 结构化prompt（中等文档，5000-10000词）
  - `create_academic_prompt_concise()`: 简洁prompt（短文档，<5000词）
  - `create_academic_prompt_detailed()`: 详细prompt（长文档，>10000词）
  - `select_prompt_by_length()`: 根据文档长度自动选择prompt
- **改进**: `create_academic_prompt()`保持向后兼容，默认使用structured风格
- **集成**: main函数支持`prompt_style: "auto"`自动选择prompt

#### `src/ensemble_predict.py`
- **更新**: `resolve_prompt_fn()`函数支持新的prompt风格（auto, structured, concise, detailed）
- **兼容性**: 保持与现有代码的完全兼容

#### `requirements.txt`
- **新增依赖**:
  - `sentence-transformers>=2.2.0`: 用于后处理中的语义相似度计算
  - `peft>=0.6.0`: 用于LoRA微调支持

### 2. 新增的文件

#### `src/postprocessing.py`
后处理模块，包含以下功能：
- `remove_duplicate_sentences()`: 使用sentence-transformers去除语义重复的句子
- `adjust_length()`: 确保摘要长度在150-220词之间（训练集平均184词）
- `ensure_coherence()`: 检查首尾句子连贯性
- `postprocess_summary()`: 主函数，整合以上三个功能

**特性**:
- 支持语义相似度检测（需要sentence-transformers）
- 如果sentence-transformers不可用，回退到精确字符串匹配
- 详细的日志输出
- 完整的类型提示和文档字符串

#### `src/chunking.py`
智能分段模块，用于处理超长文档：
- `semantic_chunking()`: 智能分段函数
  - 提取前15%（introduction）
  - 中间抽样30%
  - 提取后20%（conclusion）
  - 使用sliding window保证上下文连续性
- `chunk_document()`: 主函数，对超过12000词的文档进行分段
- 返回优化后的文本，控制在8000词以内

**策略**:
- 仅对超过12000词的文档进行分段
- 保持关键部分（开头和结尾）的完整性
- 使用滑动窗口维持上下文连续性

#### `src/quick_validate.py`
快速验证脚本，用于在小样本上快速测试模型：
- 在100条验证样本上测试模型
- 计算ROUGE-1, ROUGE-2, ROUGE-L指标
- 输出平均分、标准差、最小最大值
- 显示3个最好和最差的例子
- 支持保存详细结果到JSON文件

**功能**:
- 自动加载模型和tokenizer
- 支持多种prompt风格
- 详细的统计信息输出
- 可选的详细结果保存

#### `configs/production_config.yaml`
生产环境最佳实践配置：
- 包含所有优化的参数
- 详细的中文注释说明每个参数的作用
- `prompt_style: "auto"`（自动选择）
- `use_postprocessing: true`（启用后处理）
- `use_smart_chunking: true`（启用智能分段）
- 完整的配置选项，包括LoRA、量化等高级选项

## 如何使用新功能

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型（使用优化后的配置）

```bash
# 使用生产配置
python src/train_led_large.py --config configs/production_config.yaml

# 或使用更新后的base配置
python src/train_led_large.py --config configs/led_base_config.yaml
```

### 3. 快速验证模型

```bash
# 在100条样本上快速验证
python src/quick_validate.py \
    --config configs/production_config.yaml \
    --data data/train.csv \
    --sample_size 100 \
    --output validation_results.json
```

### 4. 使用后处理功能

```python
from src.postprocessing import postprocess_summary

# 后处理摘要
processed_summary = postprocess_summary(
    summary=raw_summary,
    remove_duplicates=True,
    adjust_len=True,
    ensure_coh=True,
    min_words=150,
    max_words=220,
)
```

### 5. 使用智能分段功能

```python
from src.chunking import chunk_document

# 对长文档进行智能分段
chunked_text = chunk_document(
    text=long_document,
    chunk_threshold=12000,
    max_words=8000,
    use_smart_chunking=True,
)
```

### 6. 集成到训练流程

在`train_led_large.py`的预测部分，可以添加后处理和分段：

```python
# 在生成预测后
from src.postprocessing import postprocess_summary
from src.chunking import chunk_document

# 对输入文本进行分段（如果启用）
if config.get("use_smart_chunking", False):
    chunked_texts = [chunk_document(text) for text in texts]

# 生成摘要后进行处理
if config.get("use_postprocessing", False):
    postprocessing_config = config.get("postprocessing", {})
    processed_summaries = [
        postprocess_summary(
            summary=summary,
            **postprocessing_config
        )
        for summary in decoded_predictions
    ]
```

## 测试

### 运行快速验证

```bash
python src/quick_validate.py \
    --config configs/production_config.yaml \
    --data data/train.csv \
    --sample_size 100
```

### 测试后处理功能

```python
# 简单测试
from src.postprocessing import postprocess_summary

test_summary = "This is a test. This is a test. This is another sentence."
result = postprocess_summary(test_summary, remove_duplicates=True)
print(result)  # 应该去除重复的句子
```

### 测试分段功能

```python
# 简单测试
from src.chunking import chunk_document

# 创建一个长文档（模拟）
long_text = " ".join(["Sentence " + str(i) + "." for i in range(10000)])
chunked = chunk_document(long_text, chunk_threshold=12000, max_words=8000)
print(f"Original: {len(long_text.split())} words")
print(f"Chunked: {len(chunked.split())} words")
```

## 配置说明

### Prompt风格选择

在配置文件中设置`prompt_style`:
- `"auto"`: 根据文档长度自动选择
  - <5000词: concise
  - 5000-10000词: structured
  - >10000词: detailed
- `"structured"`: 结构化prompt（默认）
- `"concise"`: 简洁prompt
- `"detailed"`: 详细prompt
- `"academic"`: 向后兼容，等同于structured
- `"none"`: 不使用prompt

### 后处理配置

在`production_config.yaml`中：

```yaml
use_postprocessing: true
postprocessing:
  remove_duplicates: true
  similarity_threshold: 0.85
  adjust_length: true
  min_words: 150
  max_words: 220
  ensure_coherence: true
```

### 智能分段配置

```yaml
use_smart_chunking: true
chunking:
  chunk_threshold: 12000
  max_words: 8000
  intro_ratio: 0.15
  middle_ratio: 0.30
  conclusion_ratio: 0.20
  window_size: 3
```

## 性能优化建议

1. **内存优化**: 
   - 使用`use_quantization: true`启用4-bit量化
   - 使用`gradient_checkpointing: true`减少内存占用

2. **训练速度**:
   - 调整`dataloader_num_workers`根据CPU核心数
   - 使用`fp16: true`启用混合精度训练

3. **模型质量**:
   - 使用`prompt_style: "auto"`自动选择最佳prompt
   - 启用后处理提高摘要质量
   - 对长文档启用智能分段

## 注意事项

1. **sentence-transformers**: 后处理功能需要安装`sentence-transformers`。如果未安装，会自动回退到精确字符串匹配。

2. **内存使用**: LED-Large模型较大，建议使用GPU训练。如果内存不足，可以：
   - 启用量化（`use_quantization: true`）
   - 使用LoRA（`use_lora: true`）
   - 减小`max_input_length`

3. **兼容性**: 所有新功能都保持与现有代码的向后兼容。现有的训练和预测脚本无需修改即可使用。

## 下一步

1. 在完整训练集上训练模型
2. 使用`quick_validate.py`评估模型性能
3. 根据验证结果调整超参数
4. 生成最终提交文件

