"""
TabTransformer模型实现
专门为表格数据设计的Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Q, K, V: [batch_size, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分割为多头: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力到V
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer编码块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力 + 残差连接
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TabTransformer(nn.Module):
    """
    TabTransformer模型
    用于表格数据的分类任务
    """
    
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: int = 0.1,
        embedding_dim: int = 16,
    ):
        """
        Parameters:
        -----------
        num_numeric_features : int
            数值特征的数量
        categorical_cardinalities : list
            每个分类特征的类别数（基数）
        d_model : int
            Transformer的模型维度
        num_layers : int
            Transformer层数
        num_heads : int
            注意力头数
        d_ff : int
            前馈网络维度
        dropout : float
            Dropout率
        embedding_dim : int
            分类特征嵌入维度
        """
        super().__init__()
        
        self.num_numeric_features = num_numeric_features
        self.num_categorical_features = len(categorical_cardinalities)
        self.categorical_cardinalities = categorical_cardinalities
        self.d_model = d_model
        
        # 分类特征嵌入层
        if self.num_categorical_features > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in categorical_cardinalities
            ])
            
            # 将嵌入投影到d_model
            self.categorical_projection = nn.Linear(
                self.num_categorical_features * embedding_dim,
                d_model
            )
        else:
            self.categorical_projection = None
        
        # 数值特征投影层
        if num_numeric_features > 0:
            self.numeric_projection = nn.Linear(num_numeric_features, d_model)
        else:
            self.numeric_projection = None
        
        # 位置编码（可选，对于表格数据可能不需要，但保留以增加模型表达能力）
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_categorical_features + (1 if num_numeric_features > 0 else 0), d_model)
        )
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, numeric_features=None, categorical_features=None):
        """
        Forward pass
        
        Parameters:
        -----------
        numeric_features : torch.Tensor, shape [batch_size, num_numeric_features]
            数值特征
        categorical_features : torch.Tensor, shape [batch_size, num_categorical_features]
            分类特征
        
        Returns:
        --------
        output : torch.Tensor, shape [batch_size, 1]
            预测结果
        """
        batch_size = numeric_features.size(0) if numeric_features is not None else categorical_features.size(0)
        
        # 处理分类特征
        if categorical_features is not None and self.num_categorical_features > 0:
            # 嵌入分类特征
            embedded_categorical = []
            for i, emb in enumerate(self.categorical_embeddings):
                embedded_categorical.append(emb(categorical_features[:, i].long()))
            
            # 拼接所有分类特征的嵌入
            categorical_emb = torch.cat(embedded_categorical, dim=1)  # [batch_size, num_cat * embedding_dim]
            
            # 投影到d_model
            categorical_proj = self.categorical_projection(categorical_emb)
            categorical_proj = categorical_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        else:
            categorical_proj = None
        
        # 处理数值特征
        if numeric_features is not None and self.num_numeric_features > 0:
            numeric_proj = self.numeric_projection(numeric_features)
            numeric_proj = numeric_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        else:
            numeric_proj = None
        
        # 合并特征
        features = []
        if categorical_proj is not None:
            features.append(categorical_proj)
        if numeric_proj is not None:
            features.append(numeric_proj)
        
        if len(features) == 0:
            raise ValueError("至少需要提供数值特征或分类特征之一")
        
        x = torch.cat(features, dim=1)  # [batch_size, num_features, d_model]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_features + 1, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # 通过Transformer层
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # 使用CLS token的输出进行分类
        cls_output = x[:, 0, :]  # [batch_size, d_model]
        
        # 分类
        output = self.classifier(cls_output)  # [batch_size, 1]
        
        return output


class SimpleTabTransformer(nn.Module):
    """
    简化版TabTransformer
    将数值特征和分类特征分别处理，然后融合
    """
    
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        embedding_dim: int = 32,
    ):
        super().__init__()
        
        self.num_numeric_features = num_numeric_features
        self.num_categorical_features = len(categorical_cardinalities)
        self.categorical_cardinalities = categorical_cardinalities
        
        # 分类特征嵌入和投影
        if self.num_categorical_features > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in categorical_cardinalities
            ])
            total_cat_dim = self.num_categorical_features * embedding_dim
        else:
            total_cat_dim = 0
        
        # 数值特征投影
        if num_numeric_features > 0:
            numeric_dim = num_numeric_features
        else:
            numeric_dim = 0
        
        # 特征融合层
        total_dim = total_cat_dim + numeric_dim
        self.feature_fusion = nn.Linear(total_dim, d_model)
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, numeric_features=None, categorical_features=None):
        batch_size = numeric_features.size(0) if numeric_features is not None else categorical_features.size(0)
        
        features_list = []
        
        # 处理分类特征
        if categorical_features is not None and self.num_categorical_features > 0:
            embedded_categorical = []
            for i, emb in enumerate(self.categorical_embeddings):
                embedded_categorical.append(emb(categorical_features[:, i].long()))
            cat_features = torch.cat(embedded_categorical, dim=1)
            features_list.append(cat_features)
        
        # 处理数值特征
        if numeric_features is not None and self.num_numeric_features > 0:
            features_list.append(numeric_features)
        
        # 融合特征
        if len(features_list) == 0:
            raise ValueError("至少需要提供数值特征或分类特征之一")
        
        combined_features = torch.cat(features_list, dim=1)
        x = self.feature_fusion(combined_features)
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 通过Transformer层
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # 取最后一个token的输出
        x = x.squeeze(1)  # [batch_size, d_model]
        
        # 分类
        output = self.classifier(x)
        
        return output







