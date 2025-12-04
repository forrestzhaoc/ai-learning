# Kaggle API 配置指南

## 快速配置步骤

### 1. 获取Kaggle API Token

1. 登录您的Kaggle账户：https://www.kaggle.com/
2. 点击右上角的头像，选择 **Account**
3. 滚动到 **API** 部分
4. 点击 **Create New API Token** 按钮
5. 会自动下载一个 `kaggle.json` 文件

### 2. 配置API凭证

**Linux/Mac:**
```bash
# 创建.kaggle目录
mkdir -p ~/.kaggle

# 移动kaggle.json到.kaggle目录
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置正确的权限（重要！）
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
# 创建目录
mkdir %USERPROFILE%\.kaggle

# 移动文件
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# 注意：Windows上不需要设置chmod
```

### 3. 验证配置

```bash
# 测试Kaggle API
kaggle --version
```

如果显示版本号，说明配置成功！

## 下载数据

### 方法1：使用下载脚本（推荐）

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate
python3 download_data.py
```

脚本会引导您：
1. 搜索糖尿病相关数据集
2. 选择要下载的数据集
3. 自动下载并解压

### 方法2：使用Kaggle命令行

#### 下载数据集

```bash
# 格式：kaggle datasets download -d <username>/<dataset-name>
kaggle datasets download -d uciml/pima-indians-diabetes-database
```

#### 下载竞赛数据

```bash
# 格式：kaggle competitions download -c <competition-name>
kaggle competitions download -c diabetes-prediction-challenge

# 注意：下载前需要在网站上接受竞赛规则
```

## 常见的糖尿病预测数据集

### 1. Pima Indians Diabetes Database
- **名称**: `uciml/pima-indians-diabetes-database`
- **描述**: 经典的糖尿病预测数据集
- **特征**: 8个医学特征
- **样本**: 768个样本

```bash
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip
```

### 2. Diabetes Prediction Dataset
- **名称**: `iammustafatz/diabetes-prediction-dataset`
- **描述**: 扩展的糖尿病预测数据集

```bash
kaggle datasets download -d iammustafatz/diabetes-prediction-dataset -p data --unzip
```

### 3. Diabetes Data Set
- **名称**: `mathchi/diabetes-data-set`
- **描述**: 另一个糖尿病数据集

```bash
kaggle datasets download -d mathchi/diabetes-data-set -p data --unzip
```

## 完整工作流程

### 步骤1：配置Kaggle API（仅需一次）
```bash
# 下载kaggle.json并配置
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 步骤2：下载数据
```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate

# 安装kaggle包
pip install kaggle

# 下载数据（选择一个）
python3 download_data.py
# 或者
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip
```

### 步骤3：训练模型
```bash
python3 src/train.py
```

### 步骤4：生成提交文件
```bash
python3 generate_submission.py
```

## 常见问题

### Q1: 提示"401 - Unauthorized"错误
**原因**: API凭证配置不正确

**解决方案**:
1. 检查 `~/.kaggle/kaggle.json` 是否存在
2. 检查文件权限是否为600
3. 重新下载API token

### Q2: 提示"403 - Forbidden"错误
**原因**: 
- 竞赛需要先接受规则
- 私有数据集需要权限

**解决方案**:
1. 访问Kaggle网站
2. 找到对应的竞赛/数据集
3. 点击"Join Competition"或请求访问权限

### Q3: 提示"404 - Not Found"错误
**原因**: 数据集或竞赛名称不正确

**解决方案**:
- 检查名称拼写
- 在Kaggle网站上确认正确的名称
- 使用 `kaggle datasets list -s diabetes` 搜索

### Q4: 下载速度慢
**解决方案**:
- 使用代理或VPN
- 选择较小的数据集
- 在网络较好的时间段下载

## 验证数据

下载完成后，验证数据文件：

```bash
cd data
ls -lh

# 应该看到类似以下文件：
# train.csv
# test.csv
# sample_submission.csv (可能)
```

检查数据格式：

```bash
head -5 train.csv
```

应该看到包含以下列的CSV文件：
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (仅训练集)

## 额外资源

- Kaggle API文档: https://github.com/Kaggle/kaggle-api
- Kaggle数据集搜索: https://www.kaggle.com/datasets
- Kaggle竞赛: https://www.kaggle.com/competitions

## 注意事项

1. **API配额**: Kaggle API有使用限制，不要频繁下载
2. **数据许可**: 遵守数据集的使用许可
3. **隐私**: 不要分享您的 `kaggle.json` 文件
4. **安全**: 确保 `kaggle.json` 权限设置正确（chmod 600）

## 下一步

数据下载完成后：

```bash
# 1. 查看数据
python3 src/eda.py

# 2. 训练模型
python3 src/train.py

# 3. 生成提交
python3 generate_submission.py
```

祝您在Kaggle竞赛中取得好成绩！🎉

