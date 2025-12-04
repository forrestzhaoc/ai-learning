# 🚀 开始使用：从Kaggle下载数据

## ⚡ 快速开始（3步完成）

### 第1步：配置Kaggle API

**首次使用需要配置一次，以后不需要重复**

1. 登录 https://www.kaggle.com/
2. 点击右上角头像 → Account → API → Create New API Token
3. 下载 `kaggle.json` 文件
4. 运行以下命令：

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 第2步：下载数据

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate
python3 download_data.py
```

或者直接使用命令（推荐Pima数据集）：

```bash
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip
```

### 第3步：训练并生成提交

```bash
# 训练模型（需要几分钟）
python3 src/train.py

# 生成提交文件
python3 generate_submission.py
# 选择选项2（集成模型）
```

提交文件位置：`submissions/diabetes_submission_ensemble.csv`

---

## 📚 推荐的Kaggle数据集

### 1. Pima Indians Diabetes Database ⭐ 推荐
- **数据集ID**: `uciml/pima-indians-diabetes-database`
- **特点**: 最经典的糖尿病预测数据集
- **下载命令**:
```bash
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip
```

### 2. Diabetes Prediction Dataset
- **数据集ID**: `iammustafatz/diabetes-prediction-dataset`
- **下载命令**:
```bash
kaggle datasets download -d iammustafatz/diabetes-prediction-dataset -p data --unzip
```

### 3. Diabetes Data Set
- **数据集ID**: `mathchi/diabetes-data-set`
- **下载命令**:
```bash
kaggle datasets download -d mathchi/diabetes-data-set -p data --unzip
```

---

## 🔍 验证数据

下载完成后，检查数据：

```bash
cd data
ls -lh
# 应该看到 train.csv, test.csv 等文件

# 查看数据格式
head -10 train.csv
```

期望的列：
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome（目标变量，仅训练集）

---

## 📖 详细文档

- **QUICK_KAGGLE_DOWNLOAD.md** - 快速下载指南
- **KAGGLE_SETUP.md** - 详细配置说明
- **USAGE_GUIDE.md** - 完整使用指南
- **README.md** - 项目说明

---

## ⚙️ 完整工作流程

```bash
# 1. 进入项目目录
cd /home/ubuntu/projects/ai-learning/diabetes-prediction

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 下载数据（首次需要配置API）
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip

# 4. 探索数据（可选）
python3 src/eda.py

# 5. 训练模型
python3 src/train.py

# 6. 生成提交文件
python3 generate_submission.py

# 7. 提交到Kaggle
# 上传 submissions/diabetes_submission_ensemble.csv
```

---

## 🎯 如果是Kaggle竞赛

如果您参加的是Kaggle竞赛（而不是数据集）：

```bash
# 1. 先在Kaggle网站上加入竞赛并接受规则

# 2. 下载竞赛数据
kaggle competitions download -c <competition-name> -p data

# 3. 解压文件
cd data
unzip <competition-name>.zip
cd ..

# 4. 训练和提交
python3 src/train.py
python3 generate_submission.py
```

---

## ❓ 常见问题

### Q: 提示"kaggle: command not found"
```bash
pip install kaggle
```

### Q: 提示"401 Unauthorized"
重新配置 kaggle.json：
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Q: 提示"403 Forbidden"
1. 访问数据集页面
2. 点击"Download"按钮（接受使用条款）
3. 重新运行下载命令

### Q: 数据格式不对怎么办？
确保下载的数据集包含以下列：
- 至少8个特征（Pregnancies, Glucose, BloodPressure等）
- Outcome列（训练集）

如果格式不匹配，可能需要调整 `src/data_processing.py`

---

## 🎉 成功标志

当您看到以下输出时，说明一切正常：

```
✅ 数据集下载完成！
✅ 模型训练完成！
✅ 提交文件已保存: submissions/diabetes_submission_ensemble.csv
```

---

## 📞 需要帮助？

1. 查看 `KAGGLE_SETUP.md` 了解详细配置
2. 查看 `USAGE_GUIDE.md` 了解功能详情
3. 检查代码注释了解实现细节

---

**项目位置**: `/home/ubuntu/projects/ai-learning/diabetes-prediction/`

**开始吧！祝您在Kaggle上取得好成绩！** 🏆

