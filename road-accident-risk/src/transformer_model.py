"""
Transformer-based model for Road Accident Risk Prediction
使用Transformer架构处理表格数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class TabularDataset(Dataset):
    """表格数据Dataset"""
    
    def __init__(self, features, target=None):
        # 确保数据是numpy数组且类型正确
        if isinstance(features, pd.DataFrame):
            features = features.values
        features = np.array(features, dtype=np.float32)
        self.features = torch.FloatTensor(features)
        
        if target is not None:
            if isinstance(target, pd.Series):
                target = target.values
            target = np.array(target, dtype=np.float32)
            self.target = torch.FloatTensor(target)
        else:
            self.target = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.target is not None:
            return self.features[idx], self.target[idx]
        return self.features[idx]


class TabTransformer(nn.Module):
    """
    TabTransformer: 使用Transformer处理表格数据
    
    架构:
    1. 类别特征嵌入层
    2. 数值特征直接连接
    3. Transformer Encoder处理嵌入
    4. 全连接层输出回归结果
    """
    
    def __init__(
        self,
        num_continuous_features,
        num_categorical_features,
        categorical_cardinalities,
        embedding_dim=32,
        num_heads=8,
        num_layers=4,
        hidden_dim=256,
        dropout=0.1,
        dim_feedforward=512
    ):
        super(TabTransformer, self).__init__()
        
        self.num_continuous = num_continuous_features
        self.num_categorical = num_categorical_features
        self.embedding_dim = embedding_dim
        
        # 类别特征嵌入层
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinalities
        ])
        
        # 数值特征投影层
        if num_continuous_features > 0:
            self.continuous_projection = nn.Linear(num_continuous_features, embedding_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_dim = embedding_dim * (num_categorical_features + (1 if num_continuous_features > 0 else 0))
        self.fc1 = nn.Linear(self.output_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x_continuous, x_categorical):
        """
        Forward pass
        
        Args:
            x_continuous: 数值特征 [batch_size, num_continuous]
            x_categorical: 类别特征列表 [batch_size] for each categorical feature
        """
        batch_size = x_categorical[0].size(0) if len(x_categorical) > 0 else x_continuous.size(0)
        
        # 处理类别特征嵌入
        embedded_categorical = []
        for i, cat_tensor in enumerate(x_categorical):
            embedded = self.categorical_embeddings[i](cat_tensor)
            embedded_categorical.append(embedded)
        
        # 处理数值特征
        if self.num_continuous > 0:
            continuous_embedded = self.continuous_projection(x_continuous)
            continuous_embedded = continuous_embedded.unsqueeze(1)  # [batch, 1, embedding_dim]
            embedded_categorical.append(continuous_embedded)
        
        # 拼接所有嵌入 [batch, num_features, embedding_dim]
        if len(embedded_categorical) > 0:
            x = torch.cat(embedded_categorical, dim=1)
        else:
            x = continuous_embedded
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Transformer编码
        x = self.transformer(x)  # [batch, num_features, embedding_dim]
        
        # 展平
        x = x.reshape(batch_size, -1)  # [batch, num_features * embedding_dim]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)


class SimpleTabTransformer(nn.Module):
    """
    简化版TabTransformer: 所有特征统一处理
    适用于已经编码好的数据
    """
    
    def __init__(
        self,
        input_dim,
        embedding_dim=128,
        num_heads=8,
        num_layers=6,
        hidden_dim=512,
        dropout=0.1,
        dim_feedforward=1024
    ):
        super(SimpleTabTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # 为每个特征创建独立的embedding权重
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, embedding_dim))
        
        # 位置编码（可学习的）
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embedding_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch_size, input_dim]
        """
        batch_size = x.size(0)
        
        # 投影到embedding维度
        # x: [batch, input_dim] -> [batch, input_dim, embedding_dim]
        # 将每个特征值乘以对应的embedding向量
        x = x.unsqueeze(-1)  # [batch, input_dim, 1]
        feature_emb = self.feature_embeddings.unsqueeze(0)  # [1, input_dim, embedding_dim]
        x = x * feature_emb  # [batch, input_dim, embedding_dim]
        
        # 添加位置编码（广播到batch size）
        x = x + self.pos_encoding
        
        # Transformer编码
        x = self.transformer(x)  # [batch, input_dim, embedding_dim]
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # 展平
        x = x.reshape(batch_size, -1)  # [batch, input_dim * embedding_dim]
        
        # 全连接层
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x.squeeze(-1)


def train_transformer(
    X_train, y_train,
    X_val=None, y_val=None,
    model_params=None,
    batch_size=512,
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='models/transformer_model.pth'
):
    """
    训练Transformer模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征（可选）
        y_val: 验证目标（可选）
        model_params: 模型参数
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        device: 设备
        save_path: 模型保存路径
    """
    print(f"\n{'='*60}")
    print("Training Transformer Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # 默认模型参数
    if model_params is None:
        model_params = {
            'input_dim': X_train.shape[1],
            'embedding_dim': 128,
            'num_heads': 8,
            'num_layers': 6,
            'hidden_dim': 512,
            'dropout': 0.1,
            'dim_feedforward': 1024
        }
    
    # 创建模型
    model = SimpleTabTransformer(**model_params).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据加载器
    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None and y_val is not None:
        val_dataset = TabularDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_rmse = 0.0
        
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_rmse += torch.sqrt(loss).item()
        
        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)
        
        # 验证阶段
        if X_val is not None and y_val is not None:
            model.eval()
            val_loss = 0.0
            val_rmse = 0.0
            
            with torch.no_grad():
                for batch_features, batch_target in val_loader:
                    batch_features = batch_features.to(device)
                    batch_target = batch_target.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_target)
                    
                    val_loss += loss.item()
                    val_rmse += torch.sqrt(loss).item()
            
            val_loss /= len(val_loader)
            val_rmse /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train RMSE: {train_rmse:.6f} | "
                      f"Val RMSE: {val_rmse:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_params': model_params,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        else:
            # 没有验证集，只打印训练损失
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Train RMSE: {train_rmse:.6f}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best validation RMSE: {np.sqrt(best_val_loss):.6f}")
    print(f"Model saved to: {save_path}")
    
    return model


def predict_transformer(
    X_test,
    model_path='models/transformer_model.pth',
    batch_size=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用训练好的Transformer模型进行预测
    
    Args:
        X_test: 测试特征
        model_path: 模型路径
        batch_size: 批次大小
        device: 设备
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model_params = checkpoint['model_params']
    
    model = SimpleTabTransformer(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 数据加载器
    test_dataset = TabularDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 预测
    predictions = []
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # 确保预测值非负
    predictions = np.clip(predictions, 0, None)
    
    return predictions
