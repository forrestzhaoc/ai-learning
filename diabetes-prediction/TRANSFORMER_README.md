# 🤖 Transformer 糖尿病预测模型

## 📖 简介

本项目实现了一个基于**Transformer架构**的表格数据分类模型，专门用于糖尿病预测任务。

### 特点

- ✅ **TabTransformer架构**: 专门为表格数据设计的Transformer模型
- ✅ **处理混合特征**: 同时处理数值特征和分类特征
- ✅ **注意力机制**: 使用多头自注意力学习特征间的关系
- ✅ **端到端训练**: 从原始数据到预测结果的完整流程
- ✅ **GPU加速**: 支持CUDA加速训练和推理

## 🏗️ 模型架构

### SimpleTabTransformer

```
输入数据
├── 数值特征 → 线性投影 → d_model维度
├── 分类特征 → Embedding → 拼接 → 线性投影 → d_model维度
└── 特征融合
    ↓
Transformer编码器 (3层)
├── 多头自注意力 (8头)
├── 前馈网络
└── 残差连接 + LayerNorm
    ↓
分类头
└── 输出: 糖尿病概率
```

### 关键组件

1. **分类特征嵌入层**: 将类别特征映射到连续向量空间
2. **数值特征投影层**: 将数值特征投影到统一维度
3. **Transformer编码器**: 学习特征间复杂的关系
4. **分类头**: 输出最终的预测概率

## 📦 安装依赖

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate

# 安装PyTorch (根据您的系统选择)
# CPU版本
pip install torch torchvision torchaudio

# 或CUDA版本 (如果使用GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖已在requirements.txt中
```

## 🚀 快速开始

### 1. 训练模型

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate
python3 train_transformer.py
```

**训练参数**:
- 默认使用10万样本进行训练（加速）
- Epochs: 20
- Batch size: 512
- Learning rate: 0.001
- 早停机制: 验证集AUC不再提升时自动停止

### 2. 生成提交文件

```bash
python3 generate_transformer_submission.py
```

提交文件将保存到: `submissions/transformer_submission.csv`

## 📊 模型配置

默认配置:

```python
{
    'd_model': 128,           # Transformer模型维度
    'num_layers': 3,          # Transformer层数
    'num_heads': 8,           # 注意力头数
    'd_ff': 256,              # 前馈网络维度
    'dropout': 0.1,           # Dropout率
    'embedding_dim': 32       # 分类特征嵌入维度
}
```

## 📁 文件说明

### 核心文件

- **`src/transformer_model.py`**: Transformer模型定义
  - `SimpleTabTransformer`: 简化的TabTransformer实现
  - `TabTransformer`: 完整的TabTransformer实现
  - `MultiHeadAttention`: 多头注意力机制
  - `TransformerBlock`: Transformer编码块

- **`src/data_processing_transformer.py`**: 数据处理模块
  - `TransformerDataProcessor`: 为Transformer准备数据
  - `TabularDataset`: PyTorch数据集类

- **`train_transformer.py`**: 训练脚本
- **`generate_transformer_submission.py`**: 生成提交文件

## 🔧 高级用法

### 自定义模型配置

编辑 `train_transformer.py` 中的模型创建部分:

```python
model = SimpleTabTransformer(
    num_numeric_features=num_numeric_features,
    categorical_cardinalities=categorical_cardinalities,
    d_model=256,          # 增大模型容量
    num_layers=4,         # 增加层数
    num_heads=16,         # 增加注意力头
    d_ff=512,             # 增大前馈网络
    dropout=0.2,          # 调整dropout
    embedding_dim=64      # 增大嵌入维度
)
```

### 使用全部训练数据

修改 `train_transformer.py` 中的 `train_transformer_model` 调用:

```python
model, processor, best_auc = train_transformer_model(
    train_split,
    val_split,
    sample_size=None  # 使用全部数据
)
```

### GPU训练

模型会自动检测并使用GPU（如果可用）。确保:
1. 已安装CUDA版本的PyTorch
2. 系统有可用的GPU
3. CUDA驱动已正确安装

## 📈 性能优化建议

### 1. 调整超参数

- **增大模型容量**: 增加`d_model`、`num_layers`等参数
- **调整学习率**: 尝试不同的学习率（0.0001-0.01）
- **调整batch size**: 根据GPU内存调整（128, 256, 512, 1024）

### 2. 训练策略

- **使用更多数据**: 去掉`sample_size`限制使用全部数据
- **增加训练轮数**: 调整`num_epochs`参数
- **学习率调度**: 已在代码中实现，可根据需要调整

### 3. 模型架构

- **使用完整版TabTransformer**: 替换为`TabTransformer`类
- **增加Transformer层数**: 提升模型表达能力
- **调整注意力机制**: 修改`num_heads`参数

## 🆚 与其他模型对比

| 模型 | AUC | 训练时间 | 参数量 |
|------|-----|----------|--------|
| LightGBM | 0.7130 | ~5分钟 | 小 |
| XGBoost | 0.7130 | ~5分钟 | 小 |
| Transformer | 待训练 | ~30分钟 | 中等 |

## ⚠️ 注意事项

1. **内存要求**: Transformer模型需要更多内存，建议至少8GB RAM
2. **训练时间**: 比传统机器学习模型训练时间更长
3. **数据量**: 对于小数据集，传统方法可能表现更好
4. **GPU推荐**: 使用GPU可以显著加速训练

## 🐛 常见问题

### Q: 训练速度很慢
A: 
- 检查是否使用GPU: `torch.cuda.is_available()`
- 减小batch size或使用采样数据
- 减少模型层数和维度

### Q: 内存不足
A:
- 减小batch size
- 使用采样数据训练
- 减小模型维度（d_model, d_ff）

### Q: 模型性能不如传统方法
A:
- Transformer需要更多数据才能发挥优势
- 尝试调整超参数
- 增加训练轮数
- 使用全部训练数据

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678) - TabTransformer论文
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

## 📝 项目结构

```
diabetes-prediction/
├── src/
│   ├── transformer_model.py              # Transformer模型定义
│   └── data_processing_transformer.py    # 数据处理
├── train_transformer.py                  # 训练脚本
├── generate_transformer_submission.py    # 生成提交
├── models/
│   ├── transformer_model.pth             # 训练好的模型
│   └── transformer_processor.joblib      # 数据处理器
└── submissions/
    └── transformer_submission.csv        # 提交文件
```

## 🎯 下一步

1. 训练模型: `python3 train_transformer.py`
2. 生成提交: `python3 generate_transformer_submission.py`
3. 提交到Kaggle并查看结果
4. 根据结果调整超参数和架构

---

**祝您训练顺利，取得好成绩！** 🏆





