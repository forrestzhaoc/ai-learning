# 快速从Kaggle下载数据

## 第一步：配置Kaggle API（首次使用需要）

### 1. 获取API Token
1. 访问 https://www.kaggle.com/ 并登录
2. 点击右上角头像 → **Account**
3. 滚动到 **API** 部分
4. 点击 **Create New API Token**
5. 会下载一个 `kaggle.json` 文件

### 2. 配置凭证
```bash
# 创建目录
mkdir -p ~/.kaggle

# 移动文件（假设下载到Downloads目录）
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置权限
chmod 600 ~/.kaggle/kaggle.json
```

## 第二步：下载数据

### 方法1：使用自动下载脚本（最简单）

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate
python3 download_data.py
```

然后按照提示选择数据集即可。

### 方法2：直接使用Kaggle命令

**推荐的糖尿病数据集：**

```bash
# Pima Indians Diabetes Database（最经典）
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip
```

其他选择：
```bash
# 选项2
kaggle datasets download -d iammustafatz/diabetes-prediction-dataset -p data --unzip

# 选项3
kaggle datasets download -d mathchi/diabetes-data-set -p data --unzip
```

## 第三步：验证数据

```bash
cd data
ls -lh
head -5 train.csv  # 查看前5行
```

## 第四步：开始训练

数据下载完成后：

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate

# 训练模型
python3 src/train.py

# 生成提交文件
python3 generate_submission.py
```

## 常见问题

### 问题1：提示"401 Unauthorized"
**解决**: 重新配置 kaggle.json，确保文件在 `~/.kaggle/` 目录且权限为600

### 问题2：提示"403 Forbidden"
**解决**: 
1. 访问该数据集的Kaggle页面
2. 点击"Download"或"Join Competition"接受规则

### 问题3：找不到kaggle命令
**解决**: 
```bash
pip install kaggle
```

## 一键完整流程

如果已经配置好Kaggle API：

```bash
cd /home/ubuntu/projects/ai-learning/diabetes-prediction
source venv/bin/activate

# 下载数据（选择最经典的Pima数据集）
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data --unzip

# 训练模型
python3 src/train.py

# 生成提交文件
python3 generate_submission.py
```

完成！提交文件在 `submissions/` 目录下。

## 需要帮助？

详细配置说明请查看 `KAGGLE_SETUP.md`


