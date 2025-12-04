"""
向量工具函数
用于生成节点和关系的向量嵌入
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional

# 全局模型实例（延迟加载）
_model: Optional[SentenceTransformer] = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    获取或创建嵌入模型实例
    
    Args:
        model_name: 模型名称，默认为轻量级的 all-MiniLM-L6-v2
        
    Returns:
        SentenceTransformer 模型实例
    """
    global _model
    if _model is None:
        print(f"正在加载嵌入模型: {model_name}...")
        _model = SentenceTransformer(model_name)
        print("模型加载完成！")
    return _model

def create_node_embedding(
    node_id: str,
    node_type: str,
    properties: Dict[str, Any],
    model: Optional[SentenceTransformer] = None
) -> np.ndarray:
    """
    为节点创建向量嵌入
    
    Args:
        node_id: 节点 ID
        node_type: 节点类型（如 Person, Movie）
        properties: 节点属性字典
        model: 嵌入模型，如果为 None 则使用默认模型
        
    Returns:
        节点的向量嵌入
    """
    if model is None:
        model = get_embedding_model()
    
    # 构建节点的文本表示
    text_parts = [f"类型: {node_type}"]
    
    # 添加属性到文本表示
    for key, value in properties.items():
        if value is not None:
            text_parts.append(f"{key}: {value}")
    
    text = " | ".join(text_parts)
    
    # 生成嵌入
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def create_relationship_embedding(
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None,
    model: Optional[SentenceTransformer] = None
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
    if model is None:
        model = get_embedding_model()
    
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
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

def create_query_embedding(
    query_text: str,
    model: Optional[SentenceTransformer] = None
) -> np.ndarray:
    """
    为查询文本创建向量嵌入
    
    Args:
        query_text: 查询文本
        model: 嵌入模型
        
    Returns:
        查询的向量嵌入
    """
    if model is None:
        model = get_embedding_model()
    
    embedding = model.encode(query_text, convert_to_numpy=True)
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
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    批量创建文本嵌入
    
    Args:
        texts: 文本列表
        model: 嵌入模型
        batch_size: 批处理大小
        
    Returns:
        嵌入向量数组
    """
    if model is None:
        model = get_embedding_model()
    
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return embeddings


