# Transformer模型在泰坦尼克预测中的应用

## 为什么使用Transformer？

虽然Transformer最初是为自然语言处理设计的，但也可以应用于表格数据：

1. **自动学习特征交互**：Transformer的自注意力机制可以自动发现特征之间的复杂关系
2. **处理高维特征**：通过embedding和注意力机制，可以更好地处理高维特征空间
3. **端到端学习**：不需要手动设计复杂的特征交互，模型可以自动学习

## TabTransformer架构

本项目实现的TabTransformer架构包含以下组件：

### 1. 特征处理
- **分类特征**：通过Embedding层转换为密集向量
  - 例如：Pclass, Sex, Title, Deck, AgeBin等
- **数值特征**：通过线性投影层转换为相同维度的向量
  - 例如：Age, Fare, FamilySize等

### 2. Transformer编码器
- **自注意力机制**：学习特征之间的交互关系
- **多层编码**：通过多层Transformer编码器提取深层特征
- **位置编码**：虽然表格数据没有顺序，但位置信息可以帮助模型区分不同特征

### 3. 分类头
- **特征聚合**：将所有Transformer输出拼接或池化
- **MLP分类器**：通过多层全连接网络输出最终预测

## 与树模型的对比

| 特性 | Transformer | RandomForest/XGBoost |
|------|-------------|---------------------|
| 特征交互 | 自动学习 | 需要手动设计 |
| 可解释性 | 较低 | 较高（特征重要性） |
| 训练速度 | 较慢（需要GPU） | 较快 |
| 小数据集 | 可能过拟合 | 表现稳定 |
| 大数据集 | 表现优秀 | 表现优秀 |

## 使用建议

### 何时使用Transformer：
- ✅ 数据集较大（>10k样本）
- ✅ 特征交互复杂，难以手动设计
- ✅ 有GPU资源
- ✅ 想要探索深度学习方法

### 何时使用树模型：
- ✅ 数据集较小（<10k样本）
- ✅ 需要可解释性
- ✅ 训练速度要求高
- ✅ CPU环境

## 泰坦尼克数据集的特点

泰坦尼克数据集（~900训练样本）相对较小，因此：
- **树模型**（RandomForest, XGBoost）通常表现更好
- **Transformer**可以作为实验性方法，探索深度学习的应用
- 可以通过**集成方法**结合两者优势

## 模型参数说明

```bash
python src/train_transformer.py \
  --epochs 50 \              # 训练轮数
  --batch-size 32 \           # 批次大小
  --lr 0.001 \                # 学习率
  --cv-folds 3 \              # 交叉验证折数
  --embedding-dim 32 \         # Embedding维度
  --num-layers 2 \            # Transformer层数
  --num-heads 4               # 注意力头数
```

### 参数调优建议：
- **embedding-dim**: 32-64（小数据集用较小值）
- **num-layers**: 2-4（避免过拟合）
- **num-heads**: 4-8（通常embedding_dim能被num_heads整除）
- **dropout**: 0.1-0.3（防止过拟合）

## 预期性能

对于泰坦尼克数据集：
- **RandomForest**: ~0.78-0.82 (Kaggle LB)
- **XGBoost**: ~0.78-0.83 (Kaggle LB)
- **Transformer**: ~0.75-0.80 (Kaggle LB，可能过拟合)

**注意**：由于数据集较小，Transformer可能不如树模型稳定。建议：
1. 使用更强的正则化（dropout, weight decay）
2. 减少模型复杂度（更少的层和更小的embedding维度）
3. 使用早停（early stopping）
4. 考虑集成多个模型

## 进一步优化方向

1. **特征工程**：虽然Transformer可以学习交互，但好的特征工程仍然重要
2. **正则化**：L2正则化、Dropout、Label Smoothing
3. **学习率调度**：Cosine annealing, ReduceLROnPlateau
4. **数据增强**：SMOTE, Mixup（对于表格数据）
5. **模型集成**：结合Transformer和树模型

## 参考资料

- [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)
- [FT-Transformer: Feature Tokenization + Transformer](https://arxiv.org/abs/2106.11959)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)









