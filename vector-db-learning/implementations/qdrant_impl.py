"""
Qdrant 向量数据库实现
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    CollectionStatus
)
from typing import List, Dict, Any, Optional
import numpy as np
import json
import sys
import os
import uuid
import hashlib
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

# 向量维度（使用 all-MiniLM-L6-v2 模型，维度为 384）
EMBEDDING_DIM = 384

class QdrantGraphDB:
    """使用 Qdrant 实现的图数据库接口"""
    
    def __init__(self, 
                 url: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 6333,
                 use_local: bool = True,
                 collection_prefix: str = "graph_"):
        """
        初始化 Qdrant 图数据库
        
        Args:
            url: Qdrant 服务器 URL（可选，如果提供则使用）
            host: Qdrant 服务器地址
            port: Qdrant 服务器端口
            use_local: 是否使用本地模式（内存模式，无需启动服务）
            collection_prefix: 集合名称前缀
        """
        self.collection_prefix = collection_prefix
        
        # 连接 Qdrant
        if use_local:
            # 使用本地内存模式（无需启动服务）
            self.client = QdrantClient(":memory:")
            print("✓ 已连接到 Qdrant（本地内存模式）")
        elif url:
            self.client = QdrantClient(url=url)
            print(f"✓ 已连接到 Qdrant 服务 ({url})")
        else:
            self.client = QdrantClient(host=host, port=port)
            print(f"✓ 已连接到 Qdrant 服务 ({host}:{port})")
        
        # 创建集合
        self._create_collections()
        
        # 节点元数据缓存（用于快速查找）
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        
        # 加载嵌入模型
        self.embedding_model = get_embedding_model()
    
    def _create_collections(self):
        """创建节点和关系集合"""
        nodes_collection_name = f"{self.collection_prefix}nodes"
        rels_collection_name = f"{self.collection_prefix}relationships"
        
        # 创建节点集合
        try:
            self.client.get_collection(nodes_collection_name)
            print(f"✓ 集合已存在: {nodes_collection_name}")
        except Exception:
            self.client.create_collection(
                collection_name=nodes_collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ 已创建集合: {nodes_collection_name}")
        
        # 创建关系集合
        try:
            self.client.get_collection(rels_collection_name)
            print(f"✓ 集合已存在: {rels_collection_name}")
        except Exception:
            self.client.create_collection(
                collection_name=rels_collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ 已创建集合: {rels_collection_name}")
        
        self.nodes_collection_name = nodes_collection_name
        self.relationships_collection_name = rels_collection_name
        
        # ID 映射（将原始 ID 映射到 UUID）
        self._id_to_uuid: Dict[str, str] = {}
        self._uuid_to_id: Dict[str, str] = {}
    
    def _get_uuid(self, original_id: str) -> str:
        """
        将原始 ID 转换为 UUID（保持一致性）
        
        Args:
            original_id: 原始 ID
            
        Returns:
            UUID 字符串
        """
        if original_id not in self._id_to_uuid:
            # 使用 MD5 哈希生成确定性 UUID
            hash_obj = hashlib.md5(original_id.encode())
            hex_hash = hash_obj.hexdigest()
            # 将 MD5 哈希转换为 UUID 格式
            uuid_str = f"{hex_hash[:8]}-{hex_hash[8:12]}-{hex_hash[12:16]}-{hex_hash[16:20]}-{hex_hash[20:]}"
            self._id_to_uuid[original_id] = uuid_str
            self._uuid_to_id[uuid_str] = original_id
        return self._id_to_uuid[original_id]
    
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
        
        # 准备有效载荷（payload）
        payload = {
            "node_id": node.id,
            "node_type": node.type,
            **{f"prop_{k}": str(v) for k, v in node.properties.items()}
        }
        
        # 添加到 Qdrant（使用 UUID）
        point_id = self._get_uuid(node.id)
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.nodes_collection_name,
            points=[point]
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
        
        # 准备有效载荷
        payload = {
            "rel_id": relationship.id,
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "rel_type": relationship.type,
            **{f"prop_{k}": str(v) for k, v in relationship.properties.items()}
        }
        
        # 添加到 Qdrant（使用 UUID）
        point_id = self._get_uuid(relationship.id)
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.relationships_collection_name,
            points=[point]
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
        query_filter = None
        if node_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="node_type",
                        match=MatchValue(value=node_type)
                    )
                ]
            )
        
        # 查询（使用 query_points 方法）
        results = self.client.query_points(
            collection_name=self.nodes_collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        # 格式化结果
        similar_nodes = []
        for result in results.points:
            payload = result.payload
            uuid_id = result.id
            # 将 UUID 转换回原始 ID
            node_id = self._uuid_to_id.get(uuid_id, str(uuid_id))
            
            # 从有效载荷重建节点属性
            properties = {}
            for key, value in payload.items():
                if key.startswith("prop_"):
                    prop_key = key[5:]  # 移除 "prop_" 前缀
                    properties[prop_key] = value
            
            # 获取相似度分数（如果有）
            score = getattr(result, 'score', None)
            similar_nodes.append({
                "id": node_id,
                "type": payload.get("node_type"),
                "properties": properties,
                "similarity": score
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
        # 构建过滤条件
        conditions = []
        if source_id:
            conditions.append(
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id)
                )
            )
        if target_id:
            conditions.append(
                FieldCondition(
                    key="target_id",
                    match=MatchValue(value=target_id)
                )
            )
        if rel_type:
            conditions.append(
                FieldCondition(
                    key="rel_type",
                    match=MatchValue(value=rel_type)
                )
            )
        
        query_filter = None
        if conditions:
            query_filter = Filter(must=conditions)
        
        # 使用 scroll 获取所有匹配的关系（因为我们需要按条件过滤，而不是向量搜索）
        results = self.client.scroll(
            collection_name=self.relationships_collection_name,
            scroll_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # 格式化结果
        relationships = []
        for point in results[0]:  # results 是 (points, next_page_offset) 元组
            payload = point.payload
            uuid_id = point.id
            # 将 UUID 转换回原始 ID
            rel_id = self._uuid_to_id.get(uuid_id, str(uuid_id))
            
            # 从有效载荷重建关系属性
            properties = {}
            for key, value in payload.items():
                if key.startswith("prop_"):
                    prop_key = key[5:]
                    properties[prop_key] = value
            
            relationships.append({
                "id": rel_id,
                "source_id": payload.get("source_id"),
                "target_id": payload.get("target_id"),
                "type": payload.get("rel_type"),
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
        
        # 从 Qdrant 查询（需要先转换为 UUID）
        try:
            uuid_id = self._get_uuid(node_id)
            results = self.client.retrieve(
                collection_name=self.nodes_collection_name,
                ids=[uuid_id],
                with_payload=True
            )
            
            if results and len(results) > 0:
                point = results[0]
                payload = point.payload
                
                # 从有效载荷重建节点属性
                properties = {}
                for key, value in payload.items():
                    if key.startswith("prop_"):
                        prop_key = key[5:]
                        properties[prop_key] = value
                
                node_dict = {
                    "id": node_id,
                    "type": payload.get("node_type"),
                    "properties": properties
                }
                self._node_cache[node_id] = {
                    "type": node_dict["type"],
                    "properties": node_dict["properties"]
                }
                return node_dict
        except Exception:
            pass
        
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
        try:
            self.client.delete_collection(self.nodes_collection_name)
        except Exception:
            pass
        
        try:
            self.client.delete_collection(self.relationships_collection_name)
        except Exception:
            pass
        
        self._create_collections()
        self._node_cache.clear()
        self._id_to_uuid.clear()
        self._uuid_to_id.clear()

