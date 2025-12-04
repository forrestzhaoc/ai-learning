"""
Diabetes Kaggle solution - Transformer版本（基于TabTransformer架构）

使用Transformer模型处理表格数据，通过自注意力机制学习特征交互。

用法:
    python src/train_transformer.py
    python src/train_transformer.py --epochs 50 --batch-size 32
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

# 导入数据处理模块
from data_processing_large import LargeDiabetesDataProcessor, load_and_process_data


def _create_features_helper(df):
    """创建新特征（复制LargeDiabetesDataProcessor的逻辑）"""
    df = df.copy()
    
    # BMI分类
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'], 
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=[0, 1, 2, 3]).astype(float)
    
    # 年龄分组
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'],
                                 bins=[0, 30, 40, 50, 60, 100],
                                 labels=[0, 1, 2, 3, 4]).astype(float)
    
    # 血压类别
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['bp_category'] = ((df['systolic_bp'] >= 140) | 
                            (df['diastolic_bp'] >= 90)).astype(int)
        df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    
    # 胆固醇比率
    if 'hdl_cholesterol' in df.columns and 'ldl_cholesterol' in df.columns:
        df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    
    # 生活方式评分
    if 'physical_activity_minutes_per_week' in df.columns and 'diet_score' in df.columns:
        df['lifestyle_score'] = (df['physical_activity_minutes_per_week'] / 150) * df['diet_score']
    
    # 睡眠质量指标
    if 'sleep_hours_per_day' in df.columns:
        df['sleep_quality'] = ((df['sleep_hours_per_day'] >= 7) & 
                              (df['sleep_hours_per_day'] <= 9)).astype(int)
    
    # 代谢健康指标
    if 'bmi' in df.columns and 'waist_to_hip_ratio' in df.columns:
        df['metabolic_risk'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # 心血管风险
    if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
        df['cardiovascular_risk'] = (df['heart_rate'] * df['systolic_bp']) / 10000
    
    # 风险因素计数
    risk_cols = ['family_history_diabetes', 'hypertension_history', 
                'cardiovascular_history']
    if all(col in df.columns for col in risk_cols):
        df['risk_factors_count'] = df[risk_cols].sum(axis=1)
    
    # 年龄与BMI的交互
    if 'age' in df.columns and 'bmi' in df.columns:
        df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    return df


class DiabetesDataset(Dataset):
    """糖尿病数据集"""
    def __init__(self, X_numeric, X_categorical, y=None):
        self.X_numeric = torch.FloatTensor(X_numeric)
        self.X_categorical = torch.LongTensor(X_categorical)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X_numeric)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_numeric[idx], self.X_categorical[idx], self.y[idx]
        return self.X_numeric[idx], self.X_categorical[idx]


class TabTransformer(nn.Module):
    """
    TabTransformer模型
    
    架构：
    1. 分类特征 -> Embedding -> Transformer Encoder
    2. 数值特征 -> Linear Projection
    3. 拼接 -> MLP -> 输出
    """
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        embedding_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super(TabTransformer, self).__init__()
        
        self.num_numeric_features = num_numeric_features
        self.categorical_cardinalities = categorical_cardinalities
        self.embedding_dim = embedding_dim
        
        # 分类特征Embedding层
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinalities
        ])
        
        # 数值特征投影层
        self.numeric_projection = nn.Linear(num_numeric_features, embedding_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        total_features = embedding_dim * (len(categorical_cardinalities) + 1)  # +1 for numeric
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, numeric_features, categorical_features):
        # 处理分类特征：每个特征通过embedding
        categorical_embeds = []
        for i, embedding in enumerate(self.categorical_embeddings):
            embed = embedding(categorical_features[:, i])
            categorical_embeds.append(embed)
        
        # 处理数值特征：投影到embedding维度
        numeric_embed = self.numeric_projection(numeric_features)
        
        # 拼接所有特征：shape (batch_size, num_features, embedding_dim)
        # 将数值特征也作为序列的一部分
        all_features = torch.stack([numeric_embed] + categorical_embeds, dim=1)
        
        # Transformer编码
        transformer_out = self.transformer(all_features)
        
        # 拼接所有位置的特征
        flattened = transformer_out.view(transformer_out.size(0), -1)
        
        # 分类
        output = self.classifier(flattened)
        return output


def prepare_data_for_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processor: LargeDiabetesDataProcessor,
    target_col: str = "diagnosed_diabetes",
):
    """
    准备Transformer模型所需的数据格式
    
    返回:
        X_train_numeric, X_train_categorical, y_train
        X_test_numeric, X_test_categorical
        categorical_cardinalities: 每个分类特征的类别数
    """
    # 先处理数据以获取特征工程后的DataFrame
    # 使用处理器的fit方法进行拟合
    processor.fit(train_df, target_col)
    
    # 手动处理数据以获取处理后的DataFrame（复制处理器的逻辑）
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # 编码类别变量
    for col in processor.categorical_columns:
        if col in train_processed.columns and col in processor.label_encoders:
            train_processed[col] = processor.label_encoders[col].transform(train_processed[col].astype(str))
            test_processed[col] = processor.label_encoders[col].transform(test_processed[col].astype(str))
    
    # 创建特征（复制处理器的特征工程逻辑）
    train_processed = _create_features_helper(train_processed)
    test_processed = _create_features_helper(test_processed)
    
    # 获取目标变量
    y_train = train_processed[target_col].values
    test_ids = test_processed['id'].values if 'id' in test_processed.columns else None
    
    # 定义分类特征
    categorical_feature_names = [
        'gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status',
        'bmi_category', 'age_group', 'bp_category', 'sleep_quality',
        'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
    ]
    
    # 获取所有特征列（排除id和target）
    feature_cols = [col for col in train_processed.columns 
                   if col not in [target_col, 'id']]
    
    # 只保留实际存在的分类特征
    categorical_cols = [col for col in categorical_feature_names if col in feature_cols]
    
    # 数值特征是除了分类特征之外的所有特征
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    
    # 如果还有未分类的列，根据唯一值数量判断
    remaining_cols = set(feature_cols) - set(categorical_cols) - set(numeric_cols)
    for col in remaining_cols:
        # 如果值的唯一数较少（<20），可能是分类特征
        unique_count = train_processed[col].nunique()
        if unique_count < 20:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    print(f"分类特征 ({len(categorical_cols)}): {categorical_cols}")
    print(f"数值特征 ({len(numeric_cols)}): {numeric_cols[:10]}...")
    
    # 处理数值特征：标准化
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(
        train_processed[numeric_cols].fillna(0).values
    ) if len(numeric_cols) > 0 else np.zeros((len(train_processed), 1))
    X_test_numeric = scaler.transform(
        test_processed[numeric_cols].fillna(0).values
    ) if len(numeric_cols) > 0 else np.zeros((len(test_processed), 1))
    
    # 处理分类特征：确保是连续的整数（0到cardinality-1）
    X_train_categorical_processed = []
    X_test_categorical_processed = []
    categorical_cardinalities = []
    
    for col in categorical_cols:
        # 获取唯一值数量（包括train和test）
        combined_values = pd.concat([
            train_processed[col],
            test_processed[col]
        ], ignore_index=True)
        
        # 处理NaN值
        combined_values = combined_values.fillna(-1)
        unique_values = combined_values.unique()
        cardinality = len(unique_values)
        
        # 确保值是连续的整数（0到cardinality-1）
        le = LabelEncoder()
        le.fit(combined_values.astype(str))
        
        train_values = train_processed[col].fillna(-1).astype(str)
        test_values = test_processed[col].fillna(-1).astype(str)
        
        X_train_categorical_processed.append(
            le.transform(train_values)
        )
        X_test_categorical_processed.append(
            le.transform(test_values)
        )
        
        categorical_cardinalities.append(cardinality)
    
    # 如果分类特征为空，创建一个虚拟的分类特征
    if len(categorical_cols) == 0:
        print("警告: 未找到分类特征，创建虚拟分类特征")
        X_train_categorical = np.zeros((len(train_processed), 1), dtype=np.int64)
        X_test_categorical = np.zeros((len(test_processed), 1), dtype=np.int64)
        categorical_cardinalities = [2]  # 虚拟特征有2个类别
    else:
        X_train_categorical = np.column_stack(X_train_categorical_processed).astype(np.int64)
        X_test_categorical = np.column_stack(X_test_categorical_processed).astype(np.int64)
    
    return (
        X_train_numeric, X_train_categorical, y_train,
        X_test_numeric, X_test_categorical,
        categorical_cardinalities,
        test_ids,
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for numeric, categorical, labels in dataloader:
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(numeric, categorical)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 收集预测用于AUC计算
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = correct / total
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    
    return total_loss / len(dataloader), acc, auc


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for numeric, categorical, labels in dataloader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)
            
            outputs = model(numeric, categorical)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测用于AUC计算
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = correct / total
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    
    return total_loss / len(dataloader), acc, auc


def cross_validate_transformer(
    X_numeric,
    X_categorical,
    y,
    categorical_cardinalities,
    n_splits=3,
    epochs=30,
    batch_size=32,
    lr=0.001,
    device='cpu',
):
    """交叉验证"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    cv_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_numeric, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # 准备数据
        X_train_num = X_numeric[train_idx]
        X_train_cat = X_categorical[train_idx]
        y_train = y[train_idx]
        
        X_val_num = X_numeric[val_idx]
        X_val_cat = X_categorical[val_idx]
        y_val = y[val_idx]
        
        train_dataset = DiabetesDataset(X_train_num, X_train_cat, y_train)
        val_dataset = DiabetesDataset(X_val_num, X_val_cat, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = TabTransformer(
            num_numeric_features=X_train_num.shape[1],
            categorical_cardinalities=categorical_cardinalities,
            embedding_dim=32,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            dropout=0.1,
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 训练
        best_val_auc = 0
        for epoch in range(epochs):
            train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
        
        cv_scores.append(best_val_auc)
        cv_aucs.append(best_val_auc)
        print(f"Fold {fold + 1} Best Val AUC: {best_val_auc:.4f}")
    
    return np.mean(cv_aucs), np.std(cv_aucs)


def main():
    parser = argparse.ArgumentParser(description='Diabetes Transformer Training')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--cv-folds', type=int, default=3, help='交叉验证折数')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding维度')
    parser.add_argument('--num-layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cpu/cuda/auto)')
    parser.add_argument('--sample-size', type=int, default=None, help='采样数据大小（用于快速测试）')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print("=" * 80)
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_path = os.path.join(project_dir, 'data', 'train.csv')
    test_path = os.path.join(project_dir, 'data', 'test.csv')
    
    # 加载数据
    print("读取数据...")
    if args.sample_size:
        train_df = pd.read_csv(train_path, nrows=args.sample_size)
        print(f"使用采样数据: {args.sample_size}条")
    else:
        train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 创建数据处理器
    processor = LargeDiabetesDataProcessor(scaler_type='standard')
    
    # 准备数据
    print("准备数据...")
    (
        X_train_numeric, X_train_categorical, y_train,
        X_test_numeric, X_test_categorical,
        categorical_cardinalities, test_ids,
    ) = prepare_data_for_transformer(train_df, test_df, processor)
    
    print(f"数值特征数: {X_train_numeric.shape[1]}")
    print(f"分类特征数: {len(categorical_cardinalities)}")
    print(f"分类特征类别数: {categorical_cardinalities}")
    
    # 交叉验证
    print(f"\n执行 {args.cv_folds} 折交叉验证...")
    cv_mean, cv_std = cross_validate_transformer(
        X_train_numeric,
        X_train_categorical,
        y_train,
        categorical_cardinalities,
        n_splits=args.cv_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    print(f"\nCV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # 在全量数据上训练
    print("\n在全量数据上训练最终模型...")
    train_dataset = DiabetesDataset(X_train_numeric, X_train_categorical, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = TabTransformer(
        num_numeric_features=X_train_numeric.shape[1],
        categorical_cardinalities=categorical_cardinalities,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=128,
        dropout=0.1,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for epoch in range(args.epochs):
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
    
    # 预测
    print("\n生成预测...")
    test_dataset = DiabetesDataset(X_test_numeric, X_test_categorical)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for numeric, categorical in test_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            outputs = model(numeric, categorical)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predictions.extend(probs.cpu().numpy())
    
    # 保存提交文件
    submissions_dir = os.path.join(project_dir, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)
    
    submission = pd.DataFrame({
        "id": test_ids,
        "diagnosed_diabetes": predictions,
    })
    submission_path = os.path.join(submissions_dir, "diabetes_transformer_submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存: {submission_path}")
    
    # 保存模型
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "diabetes_transformer.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'categorical_cardinalities': categorical_cardinalities,
        'num_numeric_features': X_train_numeric.shape[1],
    }, model_path)
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()

