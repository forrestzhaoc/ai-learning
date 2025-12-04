"""
ChromaDB 向量数据库实现
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试使用简化版本（不依赖网络）
try:
    from vector_utils_simple import (
        create_node_embedding,
        create_relationship_embedding,
        create_query_embedding,
        get_embedding_model
    )
except ImportError:
    from vector_utils import (
        create_node_embedding,
        create_relationship_embedding,
        create_query_embedding,
        get_embedding_model
    )
from models.graph_models import Node, Relationship

class ChromaGraphDB:
    """使用 ChromaDB 实现的图数据库接口"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化 ChromaDB 图数据库
        
        Args:
            persist_directory: 数据持久化目录
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建集合
        self.nodes_collection = self.client.get_or_create_collection(
            name="nodes",
            metadata={"description": "图节点集合"}
        )
        
        self.relationships_collection = self.client.get_or_create_collection(
            name="relationships",
            metadata={"description": "图关系集合"}
        )
        
        # 节点元数据缓存（用于快速查找）
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        
        # 加载嵌入模型
        self.embedding_model = get_embedding_model()
    
    def add_node(self, node: Node):
        """
        添加节点到数据库
        
        Args:
            node: 节点对象
        """
        # 生成节点嵌入
        embedding = create_node_embedding(
            node.id,
            node.type,
            node.properties,
            self.embedding_model
        )
        
        # 准备元数据
        metadata = {
            "node_id": node.id,
            "node_type": node.type,
            **{f"prop_{k}": str(v) for k, v in node.properties.items()}
        }
        
        # 添加到 ChromaDB
        self.nodes_collection.add(
            ids=[node.id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[f"{node.type}: {node.properties.get('name', node.id)}"]
        )
        
        # 更新缓存
        self._node_cache[node.id] = {
            "type": node.type,
            "properties": node.properties
        }
    
    def add_relationship(self, relationship: Relationship, 
                        source_node: Node, target_node: Node):
        """
        添加关系到数据库
        
        Args:
            relationship: 关系对象
            source_node: 源节点
            target_node: 目标节点
        """
        # 生成关系嵌入
        embedding = create_relationship_embedding(
            source_node.to_dict(),
            target_node.to_dict(),
            relationship.type,
            relationship.properties,
            self.embedding_model
        )
        
        # 准备元数据
        metadata = {
            "rel_id": relationship.id,
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "rel_type": relationship.type,
            **{f"prop_{k}": str(v) for k, v in relationship.properties.items()}
        }
        
        # 添加到 ChromaDB
        self.relationships_collection.add(
            ids=[relationship.id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[f"{relationship.type}: {source_node.properties.get('name', source_node.id)} -> {target_node.properties.get('name', target_node.id)}"]
        )
    
    def find_similar_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找相似节点
        
        Args:
            query: 查询文本或节点 ID
            node_type: 节点类型过滤（可选）
            top_k: 返回前 k 个结果
            
        Returns:
            相似节点列表
        """
        # 如果查询是节点 ID，先获取该节点
        if query in self._node_cache:
            node_info = self._node_cache[query]
            query_text = f"{node_info['type']}: {node_info['properties'].get('name', query)}"
        else:
            query_text = query
        
        # 生成查询嵌入
        query_embedding = create_query_embedding(query_text, self.embedding_model)
        
        # 构建过滤条件
        where = {}
        if node_type:
            where["node_type"] = node_type
        
        # 查询
        results = self.nodes_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where if where else None
        )
        
        # 格式化结果
        similar_nodes = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, node_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results else None
                
                # 从元数据重建节点属性
                properties = {}
                for key, value in metadata.items():
                    if key.startswith("prop_"):
                        prop_key = key[5:]  # 移除 "prop_" 前缀
                        properties[prop_key] = value
                
                similar_nodes.append({
                    "id": node_id,
                    "type": metadata.get("node_type"),
                    "properties": properties,
                    "similarity": 1 - distance if distance is not None else None
                })
        
        return similar_nodes
    
    def find_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找关系
        
        Args:
            source_id: 源节点 ID（可选）
            target_id: 目标节点 ID（可选）
            rel_type: 关系类型（可选）
            top_k: 返回前 k 个结果
            
        Returns:
            关系列表
        """
        # 构建过滤条件（ChromaDB 需要 $and 操作符）
        where_conditions = []
        if source_id:
            where_conditions.append({"source_id": source_id})
        if target_id:
            where_conditions.append({"target_id": target_id})
        if rel_type:
            where_conditions.append({"rel_type": rel_type})
        
        # 如果没有任何过滤条件，返回所有关系
        if not where_conditions:
            results = self.relationships_collection.get(limit=top_k)
        elif len(where_conditions) == 1:
            # 单个条件
            results = self.relationships_collection.get(
                where=where_conditions[0],
                limit=top_k
            )
        else:
            # 多个条件使用 $and
            results = self.relationships_collection.get(
                where={"$and": where_conditions},
                limit=top_k
            )
        
        # 格式化结果
        relationships = []
        if results["ids"]:
            for i, rel_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                
                # 从元数据重建关系属性
                properties = {}
                for key, value in metadata.items():
                    if key.startswith("prop_"):
                        prop_key = key[5:]
                        properties[prop_key] = value
                
                relationships.append({
                    "id": rel_id,
                    "source_id": metadata.get("source_id"),
                    "target_id": metadata.get("target_id"),
                    "type": metadata.get("rel_type"),
                    "properties": properties
                })
        
        return relationships
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取节点
        
        Args:
            node_id: 节点 ID
            
        Returns:
            节点信息，如果不存在则返回 None
        """
        if node_id in self._node_cache:
            return {
                "id": node_id,
                **self._node_cache[node_id]
            }
        
        # 从 ChromaDB 查询
        results = self.nodes_collection.get(ids=[node_id])
        if results["ids"]:
            metadata = results["metadatas"][0]
            properties = {}
            for key, value in metadata.items():
                if key.startswith("prop_"):
                    prop_key = key[5:]
                    properties[prop_key] = value
            
            node_info = {
                "id": node_id,
                "type": metadata.get("node_type"),
                "properties": properties
            }
            self._node_cache[node_id] = {
                "type": node_info["type"],
                "properties": node_info["properties"]
            }
            return node_info
        
        return None
    
    def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_hops: int = 3
    ) -> List[List[str]]:
        """
        查找两个节点之间的路径
        
        Args:
            start_node_id: 起始节点 ID
            end_node_id: 结束节点 ID
            max_hops: 最大跳数
            
        Returns:
            路径列表（每个路径是节点 ID 列表）
        """
        # 这是一个简化的路径查找实现
        # 实际应用中可以使用更复杂的图遍历算法
        
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[str], hops: int):
            if hops > max_hops:
                return
            
            if current_id == end_node_id:
                paths.append(path.copy())
                return
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            # 查找从当前节点出发的关系
            relationships = self.find_relationships(source_id=current_id)
            for rel in relationships:
                next_id = rel["target_id"]
                if next_id not in path:  # 避免循环
                    path.append(next_id)
                    dfs(next_id, path, hops + 1)
                    path.pop()
            
            visited.remove(current_id)
        
        dfs(start_node_id, [start_node_id], 0)
        return paths
    
    def clear(self):
        """清空数据库"""
        self.client.delete_collection("nodes")
        self.client.delete_collection("relationships")
        self.nodes_collection = self.client.create_collection("nodes")
        self.relationships_collection = self.client.create_collection("relationships")
        self._node_cache.clear()

