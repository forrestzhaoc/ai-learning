# Predicting Road Accident Risk

Kaggle Playground Series - Season 5, Episode 10

## 项目简介

这是一个预测道路交通事故风险严重程度的机器学习项目。基于天气条件、道路类型、时间等多种因素，预测交通事故的严重程度。

## 竞赛背景

道路交通事故是全球公共安全的重大挑战。通过机器学习模型预测事故严重程度，可以帮助：
- 改善道路安全措施
- 优化紧急响应资源分配
- 制定更有效的交通管理政策

## 项目结构

```
road-accident-risk/
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # 数据预处理和特征工程
│   ├── eda.py             # 探索性数据分析
│   └── train.py           # 模型训练
├── data/                  # 数据文件目录
├── models/                # 保存的模型
├── predictions/           # 预测结果
├── download_data.py       # 下载数据脚本
├── generate_submission.py # 生成提交文件
├── quick_start.py         # 快速开始脚本
├── requirements.txt       # 依赖包
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

首先确保你已经配置了 Kaggle API credentials (`~/.kaggle/kaggle.json`)

```bash
python download_data.py
```

### 3. 运行完整流程

```bash
python quick_start.py
```

这将自动完成：
- 数据加载和预处理
- 探索性数据分析
- 模型训练（XGBoost, LightGBM, CatBoost 集成）
- 生成提交文件

### 4. 单独运行各个步骤

**探索性数据分析：**
```bash
python -m src.eda
```

**训练模型：**
```bash
python -m src.train
```

**生成提交文件：**
```bash
python generate_submission.py
```

## 模型方法

本项目采用集成学习方法，结合三种强大的梯度提升算法：

1. **XGBoost** - 极端梯度提升，具有出色的性能和速度
2. **LightGBM** - 轻量级梯度提升机，处理大规模数据高效
3. **CatBoost** - 自动处理类别特征，减少过拟合

### 集成策略

- 使用加权平均集成多个模型的预测结果
- 通过交叉验证优化权重
- 最小化 RMSE (Root Mean Square Error)

## 特征工程

- 时间特征提取（小时、星期几、月份等）
- 天气条件编码
- 道路类型特征
- 交互特征构建
- 缺失值处理

## 评估指标

- **RMSE (Root Mean Square Error)** - 主要评估指标
- **MAE (Mean Absolute Error)** - 辅助指标
- 交叉验证分数

## 数据集

数据集来自 Kaggle Playground Series S5E10，包含：
- 训练集：包含事故特征和严重程度标签
- 测试集：需要预测事故严重程度
- 样本提交文件：提交格式参考

## 依赖环境

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- matplotlib
- seaborn

## 参考资料

- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e10)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)

## License

MIT License
