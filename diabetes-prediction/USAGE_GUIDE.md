# 糖尿病预测项目使用指南

## 项目概述

这是一个完整的Kaggle糖尿病预测竞赛解决方案，包含：
- 数据处理和特征工程
- 多种机器学习模型训练
- 模型集成
- 提交文件生成

## 快速开始

### 方法1：使用快速启动脚本（推荐）

```bash
# 进入项目目录
cd /home/ubuntu/projects/ai-learning/diabetes-prediction

# 激活虚拟环境
source venv/bin/activate

# 运行快速启动脚本
python3 quick_start.py
```

### 方法2：逐步运行

#### 1. 准备数据

如果您有真实的Kaggle数据：
```bash
# 将数据文件放入data目录
# - data/train.csv (训练集)
# - data/test.csv (测试集)
# - data/sample_submission.csv (可选)
```

如果需要创建示例数据进行测试：
```bash
source venv/bin/activate
python3 create_sample_data.py
```

#### 2. 训练模型

```bash
source venv/bin/activate
python3 src/train.py
```

这将训练以下模型：
- Logistic Regression（逻辑回归）
- Random Forest（随机森林）
- Gradient Boosting（梯度提升）
- XGBoost
- LightGBM
- SVM（支持向量机）
- Ensemble（集成模型）

#### 3. 生成提交文件

```bash
source venv/bin/activate
python3 generate_submission.py
```

选择生成方式：
- 选项1：使用最佳单模型
- 选项2：使用集成模型（推荐）
- 选项3：生成所有提交文件

## 项目结构详解

```
diabetes-prediction/
├── data/                   # 数据目录
│   ├── train.csv          # 训练数据
│   ├── test.csv           # 测试数据
│   └── sample_submission.csv
│
├── models/                 # 保存的模型
│   ├── diabetes_best_model.joblib          # 最佳单模型
│   ├── diabetes_ensemble_weights.joblib    # 集成权重
│   ├── diabetes_processor.joblib           # 数据处理器
│   ├── diabetes_logisticregression.joblib
│   ├── diabetes_randomforest.joblib
│   ├── diabetes_xgboost.joblib
│   ├── diabetes_lightgbm.joblib
│   └── model_scores.csv                    # 模型性能分数
│
├── predictions/            # 可视化和预测结果
│
├── submissions/           # 提交文件
│   ├── diabetes_submission_ensemble.csv     # 集成模型（推荐）
│   ├── diabetes_submission_best.csv         # 最佳单模型
│   ├── diabetes_submission_randomforest.csv
│   ├── diabetes_submission_xgboost.csv
│   └── ...
│
├── src/                   # 源代码
│   ├── __init__.py
│   ├── data_processing.py    # 数据处理和特征工程
│   ├── train.py             # 模型训练
│   └── eda.py              # 探索性数据分析
│
├── README.md              # 项目说明
├── USAGE_GUIDE.md        # 本文件
├── requirements.txt      # 依赖包
├── create_sample_data.py # 创建示例数据
├── generate_submission.py # 生成提交文件
└── quick_start.py        # 快速启动脚本
```

## 数据集特征说明

### 输入特征

1. **Pregnancies**: 怀孕次数
2. **Glucose**: 口服葡萄糖耐量试验中2小时血浆葡萄糖浓度
3. **BloodPressure**: 舒张压（mm Hg）
4. **SkinThickness**: 三头肌皮褶厚度（mm）
5. **Insulin**: 2小时血清胰岛素（mu U/ml）
6. **BMI**: 身体质量指数（体重kg/（身高m）^2）
7. **DiabetesPedigreeFunction**: 糖尿病遗传函数
8. **Age**: 年龄（岁）

### 目标变量

- **Outcome**: 是否患有糖尿病（0=否，1=是）

## 特征工程

项目自动创建以下派生特征：

1. **BMI_Category**: BMI分类（体重不足/正常/超重/肥胖）
2. **Age_Group**: 年龄分组
3. **BMI_Age**: BMI和年龄的交互特征
4. **Glucose_Insulin_Ratio**: 葡萄糖和胰岛素比率
5. **Pregnancies_Age_Ratio**: 怀孕次数和年龄比率
6. **BP_Age**: 血压和年龄的交互
7. **Glucose_Level**: 葡萄糖水平分类

## 模型性能

在示例数据上的模型性能（按AUC排序）：

| 模型 | 准确率 | AUC | F1-Score | 交叉验证AUC |
|------|--------|-----|----------|-------------|
| Ensemble | 0.8083 | 0.8832 | 0.7890 | - |
| RandomForest | 0.7917 | 0.8790 | 0.7863 | 0.8952 |
| GradientBoosting | 0.8000 | 0.8707 | 0.7692 | 0.8773 |
| XGBoost | 0.8250 | 0.8707 | 0.8073 | 0.8915 |
| LightGBM | 0.8167 | 0.8601 | 0.8036 | 0.8895 |
| LogisticRegression | 0.7833 | 0.8679 | 0.7636 | 0.8405 |
| SVM | 0.7750 | 0.8654 | 0.7731 | 0.8513 |

**推荐使用集成模型提交文件**，它综合了所有模型的优点。

## 进阶使用

### 探索性数据分析

```bash
source venv/bin/activate
cd src
python3 eda.py
```

这将生成：
- 特征分布图
- 相关性矩阵
- 箱线图
- 数据统计信息

### 自定义数据处理

编辑 `src/data_processing.py` 中的 `DiabetesDataProcessor` 类：

```python
processor = DiabetesDataProcessor(
    handle_zeros='median',  # 或 'mean', 'remove'
    scaler_type='standard'  # 或 'robust', None
)
```

### 调整模型参数

编辑 `src/train.py` 中的 `get_models()` 方法来修改模型超参数。

## 提交到Kaggle

1. 登录您的Kaggle账户
2. 进入糖尿病预测竞赛页面
3. 点击 "Submit Predictions"
4. 上传 `submissions/diabetes_submission_ensemble.csv`
5. 查看您的分数和排名

## 常见问题

### Q: 如何使用真实的Kaggle数据？

A: 从Kaggle下载数据后，将 `train.csv` 和 `test.csv` 放入 `data/` 目录，然后运行训练脚本。

### Q: 训练需要多长时间？

A: 在示例数据上约需2-5分钟，实际数据可能需要更长时间，取决于数据规模。

### Q: 哪个提交文件最好？

A: 推荐使用 `diabetes_submission_ensemble.csv`，它通常有最好的性能。

### Q: 如何改进模型性能？

1. 调整模型超参数
2. 创建更多特征
3. 使用更复杂的集成方法
4. 增加数据预处理步骤
5. 尝试深度学习模型

### Q: 遇到错误怎么办？

1. 确保虚拟环境已激活
2. 检查数据文件是否存在
3. 查看错误信息并根据提示修复
4. 确保所有依赖包已正确安装

## 性能优化建议

1. **数据层面**：
   - 处理缺失值和异常值
   - 进行特征缩放和归一化
   - 创建有意义的交互特征

2. **模型层面**：
   - 使用交叉验证选择最佳模型
   - 调整超参数
   - 使用集成方法

3. **评估层面**：
   - 关注多个评估指标
   - 使用K折交叉验证
   - 分析特征重要性

## 技术支持

如有问题，请查看：
- README.md - 项目基本介绍
- 代码中的注释和文档字符串
- Kaggle竞赛页面的讨论区

## 许可证

本项目仅供学习和研究使用。









