# House Prices 算法总结

本文档总结了 House Prices 竞赛中使用的不同算法及其性能。

## 算法版本

### 1. 基础版本 (train.py)

**算法**: XGBoost (单一模型)

**特点**:
- 基本特征工程
- Label encoding
- 简单的缺失值处理

**性能**:
- CV RMSE: 26,838 ± 6,022
- 训练集 RMSE: 12,760

**提交文件**: `house_prices_submission.csv`

**预测统计**:
- 价格范围: $49,380 - $512,829
- 平均价格: $178,889
- 中位数: $158,027

---

### 2. 高级 Stacking 版本 (train_advanced.py --model-type stacking)

**算法**: Stacking 集成学习

**基础模型**:
- Ridge Regression
- Lasso Regression
- ElasticNet
- XGBoost
- Random Forest
- Gradient Boosting

**元模型**: Ridge Regression

**特点**:
- 高级特征工程（114个特征）
- 对目标变量进行 log 转换
- 处理偏态分布（74个特征进行 log1p 转换）
- 异常值检测和移除
- 质量评分特征
- 多项式特征
- 交互特征
- 比例特征

**性能**:
- CV RMSE (log scale): 0.1182 ± 0.0078
- 训练集 RMSE (log scale): 0.0708

**提交文件**: `house_prices_submission_stacking_advanced.csv`

**预测统计**:
- 价格范围: $49,201 - $903,839
- 平均价格: $178,689
- 中位数: $156,168

---

### 3. 高级 Ensemble 版本 (train_advanced.py --model-type ensemble)

**算法**: 简单平均集成

**模型**:
- Ridge Regression
- Lasso Regression
- ElasticNet
- XGBoost
- Random Forest
- Gradient Boosting

**集成方法**: 简单平均

**特点**:
- 与 Stacking 版本相同的特征工程
- 更简单的集成策略

**提交文件**: `house_prices_submission_ensemble_advanced.csv`

**预测统计**:
- 价格范围: $51,529 - $879,440
- 平均价格: $178,249
- 中位数: $156,438

---

## 特征工程对比

### 基础版本特征 (89个)
- 基本数值特征
- 基本分类特征编码
- 简单的衍生特征（总面积、总浴室数等）

### 高级版本特征 (114个)
- 所有基础特征
- 质量评分特征（将 Po/Fa/TA/Gd/Ex 转换为数值）
- 年龄特征（房屋年龄、改建年龄、车库年龄）
- 二元特征（HasBasement、Has2ndFloor 等）
- 比例特征（地下室占比、生活面积占比等）
- 交互特征（质量 × 面积等）
- 多项式特征（平方、立方、平方根）
- 偏态分布处理

---

## 模型性能对比

| 模型 | CV RMSE | 价格范围 | 平均价格 | 推荐度 |
|------|---------|----------|----------|--------|
| 基础 XGBoost | 26,838 | $49K - $513K | $178,889 | ⭐⭐⭐ |
| 高级 Stacking | 0.1182 (log) | $49K - $904K | $178,689 | ⭐⭐⭐⭐⭐ |
| 高级 Ensemble | N/A | $52K - $879K | $178,249 | ⭐⭐⭐⭐ |

**推荐**: 优先使用 **高级 Stacking 版本**，因为它具有：
- 最佳的交叉验证性能
- 更全面的特征工程
- 多层集成学习
- 对目标变量进行 log 转换，减少偏态影响

---

## 使用方法

### 基础版本
```bash
python src/train.py --model-type xgb --cv-folds 5
```

### 高级 Stacking 版本
```bash
python src/train_advanced.py --model-type stacking --cv-folds 5
```

### 高级 Ensemble 版本
```bash
python src/train_advanced.py --model-type ensemble --cv-folds 5
```

---

## 文件位置

### 提交文件
- `submissions/house_prices_submission.csv` - 基础版本
- `submissions/house_prices_submission_stacking_advanced.csv` - 高级 Stacking
- `submissions/house_prices_submission_ensemble_advanced.csv` - 高级 Ensemble

### 模型文件
- `models/house_prices_model.joblib` - 基础版本
- `models/house_prices_stacking_advanced.joblib` - 高级 Stacking
- `models/house_prices_ensemble_advanced.joblib` - 高级 Ensemble

### 预测结果
- `predictions/house_prices_predictions_xgb.csv` - 基础版本
- `predictions/house_prices_predictions_stacking_advanced.csv` - 高级 Stacking
- `predictions/house_prices_predictions_ensemble_advanced.csv` - 高级 Ensemble

---

## 改进建议

1. **超参数优化**: 使用 Optuna 或 GridSearchCV 进行更系统的超参数搜索
2. **特征选择**: 使用特征重要性分析移除冗余特征
3. **更多模型**: 尝试 LightGBM、CatBoost
4. **深度学习**: 尝试神经网络方法
5. **特征工程**: 探索更多领域知识相关的特征

---

生成日期: 2025-12-01

