# Diabetes Prediction Challenge

这是一个Kaggle糖尿病预测竞赛的完整解决方案。

## 项目结构

```
diabetes-prediction/
├── data/               # 数据文件目录
│   ├── train.csv      # 训练数据
│   ├── test.csv       # 测试数据
│   └── sample_submission.csv
├── models/            # 保存的模型文件
├── predictions/       # 预测结果
├── submissions/       # 提交文件
├── src/              # 源代码
│   ├── data_processing.py    # 数据处理
│   ├── train.py             # 模型训练
│   └── eda.py              # 探索性数据分析
├── requirements.txt   # 依赖包
└── generate_submission.py  # 生成提交文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将从Kaggle下载的数据文件放到 `data/` 目录:
- train.csv
- test.csv
- sample_submission.csv

### 3. 训练模型

```bash
python src/train.py
```

### 4. 生成提交文件

```bash
python generate_submission.py
```

提交文件将保存在 `submissions/` 目录中。

## 特征说明

数据集通常包含以下特征：
- **Pregnancies**: 怀孕次数
- **Glucose**: 血糖浓度
- **BloodPressure**: 血压
- **SkinThickness**: 皮肤厚度
- **Insulin**: 胰岛素水平
- **BMI**: 身体质量指数
- **DiabetesPedigreeFunction**: 糖尿病遗传函数
- **Age**: 年龄
- **Outcome**: 目标变量（是否患有糖尿病，0或1）

## 模型方法

本项目使用多种算法进行预测：
1. Logistic Regression（逻辑回归）
2. Random Forest（随机森林）
3. XGBoost
4. LightGBM
5. Ensemble（集成模型）

## 评估指标

常用的评估指标包括：
- Accuracy（准确率）
- AUC-ROC
- F1 Score
- Precision/Recall

## 注意事项

- 确保数据文件格式正确
- 注意处理缺失值和异常值
- 使用交叉验证评估模型性能
- 提交文件格式必须与sample_submission.csv一致

