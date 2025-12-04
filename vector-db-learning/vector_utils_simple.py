"""
简化的向量工具函数（不依赖网络）
使用简单的哈希和特征提取方法生成向量
"""

import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import json

# 全局模型实例（延迟加载）
_model = None

def get_embedding_model(model_name: str = "simple"):
    """
    获取或创建简化的嵌入模型实例（不依赖网络）
    
    Args:
        model_name: 模型名称（忽略，使用简单方法）
        
    Returns:
        模型对象（实际是 None，但保持接口一致）
    """
    global _model
    if _model is None:
        print("使用简化的向量生成方法（无需网络）...")
        _model = "simple"
    return _model

def text_to_vector(text: str, dim: int = 384) -> np.ndarray:
    """
    将文本转换为向量（使用哈希和特征提取）
    
    Args:
        text: 输入文本
        dim: 向量维度
        
    Returns:
        向量嵌入
    """
    # 使用多种哈希方法组合生成向量
    vector = np.zeros(dim)
    
    # 方法1: 基于字符的哈希
    for i, char in enumerate(text[:100]):  # 限制长度
        hash_val = hash(char + str(i)) % dim
        vector[hash_val] += 1.0
    
    # 方法2: 基于单词的哈希
    words = text.lower().split()
    for word in words[:50]:  # 限制单词数
        hash_val = hash(word) % dim
        vector[hash_val] += 0.5
    
    # 方法3: 基于文本长度的特征
    text_len = len(text)
    for i in range(min(10, dim)):
        vector[i] += (text_len % (i + 1)) / 100.0
    
    # 归一化
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.astype(np.float32)

def create_node_embedding(
    node_id: str,
    node_type: str,
    properties: Dict[str, Any],
    model: Optional[Any] = None
) -> np.ndarray:
    """
    为节点创建向量嵌入
    
    Args:
        node_id: 节点 ID
        node_type: 节点类型（如 Person, Movie）
        properties: 节点属性字典
        model: 嵌入模型（忽略）
        
    Returns:
        节点的向量嵌入
    """
    # 构建节点的文本表示
    text_parts = [f"类型: {node_type}"]
    
    # 添加属性到文本表示
    for key, value in properties.items():
        if value is not None:
            text_parts.append(f"{key}: {value}")
    
    text = " | ".join(text_parts)
    
    # 生成嵌入
    embedding = text_to_vector(text)
    return embedding

def create_relationship_embedding(
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None
) -> np.ndarray:
    """
    为关系创建向量嵌入
    
    Args:
        source_node: 源节点信息（包含 id, type, properties）
        target_node: 目标节点信息（包含 id, type, properties）
        relationship_type: 关系类型（如 ACTED_IN, DIRECTED）
        properties: 关系属性字典
        model: 嵌入模型
        
    Returns:
        关系的向量嵌入
    """
    # 构建关系的文本表示
    text_parts = [
        f"关系类型: {relationship_type}",
        f"源节点: {source_node.get('type', 'Unknown')} - {source_node.get('properties', {}).get('name', source_node.get('id', ''))}",
        f"目标节点: {target_node.get('type', 'Unknown')} - {target_node.get('properties', {}).get('name', target_node.get('id', ''))}"
    ]
    
    if properties:
        for key, value in properties.items():
            if value is not None:
                text_parts.append(f"{key}: {value}")
    
    text = " | ".join(text_parts)
    
    # 生成嵌入
    embedding = text_to_vector(text)
    return embedding

def create_query_embedding(
    query_text: str,
    model: Optional[Any] = None
) -> np.ndarray:
    """
    为查询文本创建向量嵌入
    
    Args:
        query_text: 查询文本
        model: 嵌入模型
        
    Returns:
        查询的向量嵌入
    """
    embedding = text_to_vector(query_text)
    return embedding

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        余弦相似度值（-1 到 1）
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def batch_create_embeddings(
    texts: List[str],
    model: Optional[Any] = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    批量创建文本嵌入
    
    Args:
        texts: 文本列表
        model: 嵌入模型
        batch_size: 批处理大小（忽略）
        
    Returns:
        嵌入向量数组
    """
    embeddings = np.array([text_to_vector(text) for text in texts])
    return embeddings



