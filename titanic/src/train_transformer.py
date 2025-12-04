"""
Titanic Kaggle solution - Transformer版本（基于TabTransformer架构）

使用Transformer模型处理表格数据，通过自注意力机制学习特征交互。

用法:
    python src/train_transformer.py
    python src/train_transformer.py --epochs 50 --batch-size 32
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import accuracy_score

# 复用特征工程代码
from train_simple import (
    MANDATORY_COLS,
    AGE_BINS,
    AGE_LABELS,
    extract_title,
    _fill_missing,
    engineer_features,
    encode_and_align,
    add_interactions,
    build_datasets,
)


class TitanicDataset(Dataset):
    """泰坦尼克数据集"""
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
        
        # 全局平均池化 + 拼接
        # 方式1：使用CLS token（第一个位置，即数值特征）
        # 方式2：平均池化所有位置
        pooled = transformer_out.mean(dim=1)  # (batch_size, embedding_dim)
        
        # 或者拼接所有位置的特征
        flattened = transformer_out.view(transformer_out.size(0), -1)
        
        # 分类
        output = self.classifier(flattened)
        return output


def prepare_data_for_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Survived",
):
    """
    准备Transformer模型所需的数据格式
    
    返回:
        X_train_numeric, X_train_categorical, y_train
        X_test_numeric, X_test_categorical
        categorical_cardinalities: 每个分类特征的类别数
    """
    # 特征工程（复用现有代码）
    X_train, X_test = build_datasets(train_df, test_df)
    y_train = train_df[target_col].values
    
    # 定义分类特征（即使被编码成整数，也应该作为分类特征）
    # 这些特征有明确的类别含义，应该用embedding
    categorical_feature_names = [
        'Pclass',  # 1, 2, 3
        'Sex',  # 0, 1
        'Title',  # 编码后的类别
        'Deck',  # 编码后的类别
        'AgeBin',  # 编码后的类别
        'TicketPrefix',  # 编码后的类别
        'FareBin',  # 分箱后的类别
        'Embarked_C', 'Embarked_Q', 'Embarked_S',  # one-hot编码
    ]
    
    # 数值特征（连续值）
    numeric_feature_names = [
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'FamilySize',
        'IsAlone',
        'FarePerPerson',
        'HasCabin',
        'CabinCount',
        'TicketGroupSize',
        # 交互特征
        'IsAlone_Sex',
        'IsAlone_Pclass',
        'IsAlone_Age',
        'Pclass_Sex',
        'Pclass_Fare',
        'Pclass_Age',
        'Pclass_IsAlone',
        'Title_Sex',
        'Title_Pclass',
        'Title_IsAlone',
        'Title_Fare',
        'AgeBin_Pclass',
        'FareBin_Pclass',
    ]
    
    # 只保留实际存在的列
    categorical_cols = [col for col in categorical_feature_names if col in X_train.columns]
    numeric_cols = [col for col in numeric_feature_names if col in X_train.columns]
    
    # 如果还有未分类的列，根据数据类型判断
    remaining_cols = set(X_train.columns) - set(categorical_cols) - set(numeric_cols)
    for col in remaining_cols:
        if col in ['PassengerId']:
            continue  # 跳过ID列
        # 如果值的唯一数较少（<20），可能是分类特征
        if X_train[col].nunique() < 20:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    print(f"分类特征 ({len(categorical_cols)}): {categorical_cols[:10]}...")
    print(f"数值特征 ({len(numeric_cols)}): {numeric_cols[:10]}...")
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[numeric_cols].fillna(0))
    X_test_numeric = scaler.transform(X_test[numeric_cols].fillna(0))
    
    # 处理分类特征：确保是整数类型
    X_train_categorical = pd.DataFrame()
    X_test_categorical = pd.DataFrame()
    categorical_cardinalities = []
    
    for col in categorical_cols:
        if col in X_train.columns:
            # 获取唯一值数量（包括train和test）
            combined_values = pd.concat([X_train[col], X_test[col]], ignore_index=True)
            unique_values = combined_values.fillna(-1).unique()
            cardinality = len(unique_values)
            
            # 确保值是连续的整数（0到cardinality-1）
            le = LabelEncoder()
            combined_encoded = le.fit_transform(combined_values.fillna('Missing'))
            
            X_train_categorical[col] = le.transform(X_train[col].fillna('Missing'))
            X_test_categorical[col] = le.transform(X_test[col].fillna('Missing'))
            
            categorical_cardinalities.append(cardinality)
    
    # 如果没有分类特征，使用Pclass作为分类特征
    if len(categorical_cols) == 0:
        print("警告: 未找到分类特征，使用Pclass作为分类特征")
        categorical_cols = ['Pclass']
        le = LabelEncoder()
        combined = pd.concat([X_train['Pclass'], X_test['Pclass']], ignore_index=True)
        X_train_categorical['Pclass'] = le.fit_transform(X_train['Pclass'])
        X_test_categorical['Pclass'] = le.transform(X_test['Pclass'])
        categorical_cardinalities = [len(le.classes_)]
    
    X_train_categorical = X_train_categorical.values.astype(np.int64)
    X_test_categorical = X_test_categorical.values.astype(np.int64)
    
    return (
        X_train_numeric, X_train_categorical, y_train,
        X_test_numeric, X_test_categorical,
        categorical_cardinalities,
        scaler,
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
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
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
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
    
    return total_loss / len(dataloader), correct / total


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
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_numeric, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # 准备数据
        X_train_num = X_numeric[train_idx]
        X_train_cat = X_categorical[train_idx]
        y_train = y[train_idx]
        
        X_val_num = X_numeric[val_idx]
        X_val_cat = X_categorical[val_idx]
        y_val = y[val_idx]
        
        train_dataset = TitanicDataset(X_train_num, X_train_cat, y_train)
        val_dataset = TitanicDataset(X_val_num, X_val_cat, y_val)
        
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
        best_val_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        cv_scores.append(best_val_acc)
        print(f"Fold {fold + 1} Best Val Acc: {best_val_acc:.4f}")
    
    return np.mean(cv_scores), np.std(cv_scores)


def main():
    parser = argparse.ArgumentParser(description='Titanic Transformer Training')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--cv-folds', type=int, default=3, help='交叉验证折数')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding维度')
    parser.add_argument('--num-layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print("=" * 80)
    
    # 加载数据
    print("读取数据...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # 准备数据
    print("准备数据...")
    (
        X_train_numeric, X_train_categorical, y_train,
        X_test_numeric, X_test_categorical,
        categorical_cardinalities, scaler,
    ) = prepare_data_for_transformer(train_df, test_df)
    
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
    print(f"\nCV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # 在全量数据上训练
    print("\n在全量数据上训练最终模型...")
    train_dataset = TitanicDataset(X_train_numeric, X_train_categorical, y_train)
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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step(train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    
    # 预测
    print("\n生成预测...")
    test_dataset = TitanicDataset(X_test_numeric, X_test_categorical)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for numeric, categorical in test_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            outputs = model(numeric, categorical)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # 保存提交文件
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions,
    })
    submission.to_csv("submissions/titanic_submission_transformer.csv", index=False)
    print("提交文件已保存: submissions/titanic_submission_transformer.csv")
    
    # 保存模型
    model_path = Path("models/titanic_transformer.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'categorical_cardinalities': categorical_cardinalities,
        'num_numeric_features': X_train_numeric.shape[1],
    }, model_path)
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()

