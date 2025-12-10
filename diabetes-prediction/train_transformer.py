"""
使用Transformer训练糖尿病预测模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from transformer_model import SimpleTabTransformer
from data_processing_transformer import TransformerDataProcessor, create_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        numeric = batch['numeric'].to(device)
        categorical = batch['categorical'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(numeric_features=numeric, categorical_features=categorical)
        outputs = outputs.squeeze()
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测结果
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    preds_binary = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, preds_binary)
    
    return avg_loss, accuracy, auc, f1


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(numeric_features=numeric, categorical_features=categorical)
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    preds_binary = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, preds_binary)
    
    return avg_loss, accuracy, auc, f1


def train_transformer_model(
    train_df,
    val_df,
    num_epochs=20,
    batch_size=256,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    sample_size=None
):
    """训练Transformer模型"""
    print("=" * 70)
    print("训练TabTransformer模型")
    print("=" * 70)
    
    # 使用采样数据加速（如果指定）
    if sample_size and len(train_df) > sample_size:
        print(f"\n使用采样数据: {sample_size} 条")
        train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"\n训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 创建数据处理器
    processor = TransformerDataProcessor()
    
    # 创建数据加载器
    print("\n准备数据...")
    train_loader, val_loader, processor = create_data_loaders(
        train_df,
        val_df,
        processor,
        batch_size=batch_size
    )
    
    # 获取模型参数
    num_numeric_features = len(processor.numeric_feature_names)
    categorical_cardinalities = processor.categorical_cardinalities
    
    print(f"\n模型参数:")
    print(f"  数值特征数: {num_numeric_features}")
    print(f"  分类特征数: {len(categorical_cardinalities)}")
    print(f"  分类特征基数: {categorical_cardinalities}")
    
    # 创建模型
    model = SimpleTabTransformer(
        num_numeric_features=num_numeric_features,
        categorical_cardinalities=categorical_cardinalities,
        d_model=128,
        num_layers=3,
        num_heads=8,
        d_ff=256,
        dropout=0.1,
        embedding_dim=32
    )
    
    model = model.to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数数量:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print(f"\n开始训练 ({num_epochs} epochs)...")
    print(f"设备: {device}")
    print("-" * 70)
    
    best_auc = 0
    best_model_state = None
    patience_counter = 0
    patience = 5
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc, train_auc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc, val_auc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ 新的最佳模型 (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发，停止训练")
                break
        
        print("-" * 70)
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n最佳验证集AUC: {best_auc:.4f}")
    
    return model, processor, best_auc


def main():
    """主函数"""
    print("=" * 70)
    print("TabTransformer 糖尿病预测模型训练")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv('data/train.csv')
    print(f"训练集形状: {train_df.shape}")
    
    # 分割训练集和验证集
    print("\n分割训练集和验证集...")
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['diagnosed_diabetes']
    )
    
    print(f"训练集: {len(train_split)}")
    print(f"验证集: {len(val_split)}")
    
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 训练模型（使用全部数据）
    print(f"\n使用全部训练数据: {len(train_split)} 样本")
    print("这将需要更长的训练时间，但可能获得更好的性能")
    
    model, processor, best_auc = train_transformer_model(
        train_split,
        val_split,
        num_epochs=20,
        batch_size=512,
        learning_rate=0.001,
        device=device,
        sample_size=None  # 使用全部数据训练
    )
    
    # 保存模型
    print("\n" + "=" * 70)
    print("保存模型")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # 保存模型
    model_path = 'models/transformer_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_auc': best_auc,
        'num_numeric_features': len(processor.numeric_feature_names),
        'categorical_cardinalities': processor.categorical_cardinalities,
        'model_config': {
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 8,
            'd_ff': 256,
            'dropout': 0.1,
            'embedding_dim': 32
        }
    }, model_path)
    print(f"已保存模型: {model_path}")
    
    # 保存数据处理器
    processor_path = 'models/transformer_processor.joblib'
    joblib.dump(processor, processor_path)
    print(f"已保存数据处理器: {processor_path}")
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"最佳验证集AUC: {best_auc:.4f}")
    print("\n接下来运行: python3 generate_transformer_submission.py")


if __name__ == '__main__':
    main()

