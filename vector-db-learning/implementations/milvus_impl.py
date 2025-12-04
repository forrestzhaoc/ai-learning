"""
Milvus 向量数据库实现
"""

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_utils import (
    create_node_embedding,
    create_relationship_embedding,
    create_query_embedding,
    get_embedding_model
)
from models.graph_models import Node, Relationship

# 向量维度（使用 all-MiniLM-L6-v2 模型，维度为 384）
EMBEDDING_DIM = 384

class MilvusGraphDB:
    """使用 Milvus 实现的图数据库接口"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 19530,
                 use_lite: bool = True,
                 collection_prefix: str = "graph_"):
        """
        初始化 Milvus 图数据库
        
        Args:
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            use_lite: 是否使用 Milvus Lite（本地模式，无需启动服务）
            collection_prefix: 集合名称前缀
        """
        self.collection_prefix = collection_prefix
        self.use_lite = use_lite
        
        # 连接 Milvus
        if use_lite:
            try:
                # 尝试使用 Milvus Lite (MilvusClient)
                from pymilvus import MilvusClient
                self.client = MilvusClient()
                self.use_lite_client = True
                print("✓ 已连接到 Milvus Lite（本地模式）")
            except (ImportError, Exception) as e:
                print(f"⚠ Milvus Lite 初始化失败: {e}")
                print("⚠ 尝试连接远程 Milvus 服务...")
                try:
                    connections.connect("default", host=host, port=port)
                    self.use_lite_client = False
                    print(f"✓ 已连接到 Milvus 服务 ({host}:{port})")
                except Exception as e2:
                    raise Exception(f"无法连接到 Milvus: {e2}")
        else:
            connections.connect("default", host=host, port=port)
            self.use_lite_client = False
            print(f"✓ 已连接到 Milvus 服务 ({host}:{port})")
        
        # 创建集合
        self._create_collections()
        
        # 节点元数据缓存（用于快速查找）
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        
        # 加载嵌入模型
        self.embedding_model = get_embedding_model()
    
    def _create_collections(self):
        """创建节点和关系集合"""
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            # 使用 MilvusClient（Lite 模式）
            self._create_collections_lite()
        else:
            # 使用标准 Milvus 连接
            self._create_collections_standard()
    
    def _create_collections_lite(self):
        """使用 MilvusClient 创建集合（Lite 模式）"""
        nodes_collection_name = f"{self.collection_prefix}nodes"
        rels_collection_name = f"{self.collection_prefix}relationships"
        
        # 检查集合是否存在
        if nodes_collection_name not in self.client.list_collections():
            self.client.create_collection(
                collection_name=nodes_collection_name,
                dimension=EMBEDDING_DIM,
                description="图节点集合"
            )
            print(f"✓ 已创建集合: {nodes_collection_name}")
        else:
            print(f"✓ 集合已存在: {nodes_collection_name}")
        
        if rels_collection_name not in self.client.list_collections():
            self.client.create_collection(
                collection_name=rels_collection_name,
                dimension=EMBEDDING_DIM,
                description="图关系集合"
            )
            print(f"✓ 已创建集合: {rels_collection_name}")
        else:
            print(f"✓ 集合已存在: {rels_collection_name}")
        
        self.nodes_collection_name = nodes_collection_name
        self.relationships_collection_name = rels_collection_name
    
    def _create_collections_standard(self):
        """使用标准 Milvus 连接创建集合"""
        nodes_collection_name = f"{self.collection_prefix}nodes"
        rels_collection_name = f"{self.collection_prefix}relationships"
        
        # 节点集合 schema
        nodes_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="properties", dtype=DataType.VARCHAR, max_length=2000),
        ]
        nodes_schema = CollectionSchema(
            fields=nodes_fields,
            description="图节点集合"
        )
        
        # 关系集合 schema
        rels_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="target_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="rel_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="properties", dtype=DataType.VARCHAR, max_length=2000),
        ]
        rels_schema = CollectionSchema(
            fields=rels_fields,
            description="图关系集合"
        )
        
        # 创建集合
        if utility.has_collection(nodes_collection_name):
            self.nodes_collection = Collection(nodes_collection_name)
            print(f"✓ 集合已存在: {nodes_collection_name}")
        else:
            self.nodes_collection = Collection(nodes_collection_name, schema=nodes_schema)
            print(f"✓ 已创建集合: {nodes_collection_name}")
        
        if utility.has_collection(rels_collection_name):
            self.relationships_collection = Collection(rels_collection_name)
            print(f"✓ 集合已存在: {rels_collection_name}")
        else:
            self.relationships_collection = Collection(rels_collection_name, schema=rels_schema)
            print(f"✓ 已创建集合: {rels_collection_name}")
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        if not self.nodes_collection.has_index():
            self.nodes_collection.create_index("embedding", index_params)
        
        if not self.relationships_collection.has_index():
            self.relationships_collection.create_index("embedding", index_params)
        
        # 加载集合到内存
        self.nodes_collection.load()
        self.relationships_collection.load()
        
        self.nodes_collection_name = nodes_collection_name
        self.relationships_collection_name = rels_collection_name
    
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
        
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            # 使用 MilvusClient
            data = [{
                "id": node.id,
                "vector": embedding.tolist(),
                "node_type": node.type,
                "properties": json.dumps(node.properties, ensure_ascii=False)
            }]
            self.client.insert(
                collection_name=self.nodes_collection_name,
                data=data
            )
        else:
            # 使用标准 Milvus
            data = [[node.id], [embedding.tolist()], [node.type], [json.dumps(node.properties, ensure_ascii=False)]]
            self.nodes_collection.insert(data)
        
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
        
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            # 使用 MilvusClient
            data = [{
                "id": relationship.id,
                "vector": embedding.tolist(),
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "rel_type": relationship.type,
                "properties": json.dumps(relationship.properties, ensure_ascii=False)
            }]
            self.client.insert(
                collection_name=self.relationships_collection_name,
                data=data
            )
        else:
            # 使用标准 Milvus
            data = [
                [relationship.id],
                [embedding.tolist()],
                [relationship.source_id],
                [relationship.target_id],
                [relationship.type],
                [json.dumps(relationship.properties, ensure_ascii=False)]
            ]
            self.relationships_collection.insert(data)
    
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
        
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            # 使用 MilvusClient 搜索
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = self.client.search(
                collection_name=self.nodes_collection_name,
                data=[query_embedding.tolist()],
                limit=top_k * 2,  # 多取一些，然后过滤
                search_params=search_params
            )
            
            # 处理结果
            similar_nodes = []
            if results and len(results) > 0:
                for hit in results[0]:
                    node_id = hit["id"]
                    distance = hit.get("distance", 0)
                    
                    # 获取节点详细信息
                    node_data = self.client.get(
                        collection_name=self.nodes_collection_name,
                        ids=[node_id]
                    )
                    
                    if node_data and len(node_data) > 0:
                        node_info = node_data[0]
                        node_type_found = node_info.get("node_type", "")
                        
                        # 类型过滤
                        if node_type and node_type_found != node_type:
                            continue
                        
                        properties = json.loads(node_info.get("properties", "{}"))
                        
                        similar_nodes.append({
                            "id": node_id,
                            "type": node_type_found,
                            "properties": properties,
                            "similarity": 1 / (1 + distance) if distance > 0 else 1.0  # 转换为相似度
                        })
                        
                        if len(similar_nodes) >= top_k:
                            break
        else:
            # 使用标准 Milvus 搜索
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = self.nodes_collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2,
                output_fields=["node_type", "properties"]
            )
            
            # 处理结果
            similar_nodes = []
            if results and len(results) > 0:
                for hit in results[0]:
                    node_id = hit.id
                    distance = hit.distance
                    node_type_found = hit.entity.get("node_type", "")
                    
                    # 类型过滤
                    if node_type and node_type_found != node_type:
                        continue
                    
                    properties = json.loads(hit.entity.get("properties", "{}"))
                    
                    similar_nodes.append({
                        "id": node_id,
                        "type": node_type_found,
                        "properties": properties,
                        "similarity": 1 / (1 + distance) if distance > 0 else 1.0
                    })
                    
                    if len(similar_nodes) >= top_k:
                        break
        
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
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            # 使用 MilvusClient 查询
            # 注意：MilvusClient 的过滤查询需要先获取所有数据然后过滤
            # 这里简化实现，获取所有关系然后过滤
            all_data = self.client.query(
                collection_name=self.relationships_collection_name,
                filter="",  # 空过滤获取所有
                limit=1000  # 限制数量
            )
            
            relationships = []
            for item in all_data:
                # 应用过滤条件
                if source_id and item.get("source_id") != source_id:
                    continue
                if target_id and item.get("target_id") != target_id:
                    continue
                if rel_type and item.get("rel_type") != rel_type:
                    continue
                
                properties = json.loads(item.get("properties", "{}"))
                relationships.append({
                    "id": item.get("id"),
                    "source_id": item.get("source_id"),
                    "target_id": item.get("target_id"),
                    "type": item.get("rel_type"),
                    "properties": properties
                })
                
                if len(relationships) >= top_k:
                    break
        else:
            # 使用标准 Milvus 查询
            # 构建过滤表达式
            filter_expr = []
            if source_id:
                filter_expr.append(f'source_id == "{source_id}"')
            if target_id:
                filter_expr.append(f'target_id == "{target_id}"')
            if rel_type:
                filter_expr.append(f'rel_type == "{rel_type}"')
            
            filter_str = " && ".join(filter_expr) if filter_expr else ""
            
            results = self.relationships_collection.query(
                expr=filter_str if filter_str else "id != ''",
                limit=top_k,
                output_fields=["source_id", "target_id", "rel_type", "properties"]
            )
            
            relationships = []
            for item in results:
                properties = json.loads(item.get("properties", "{}"))
                relationships.append({
                    "id": item.get("id"),
                    "source_id": item.get("source_id"),
                    "target_id": item.get("target_id"),
                    "type": item.get("rel_type"),
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
        
        # 从 Milvus 查询
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            node_data = self.client.get(
                collection_name=self.nodes_collection_name,
                ids=[node_id]
            )
            if node_data and len(node_data) > 0:
                node_info = node_data[0]
                properties = json.loads(node_info.get("properties", "{}"))
                node_dict = {
                    "id": node_id,
                    "type": node_info.get("node_type"),
                    "properties": properties
                }
                self._node_cache[node_id] = {
                    "type": node_dict["type"],
                    "properties": node_dict["properties"]
                }
                return node_dict
        else:
            results = self.nodes_collection.query(
                expr=f'id == "{node_id}"',
                output_fields=["node_type", "properties"]
            )
            if results and len(results) > 0:
                node_info = results[0]
                properties = json.loads(node_info.get("properties", "{}"))
                node_dict = {
                    "id": node_id,
                    "type": node_info.get("node_type"),
                    "properties": properties
                }
                self._node_cache[node_id] = {
                    "type": node_dict["type"],
                    "properties": node_dict["properties"]
                }
                return node_dict
        
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
        if hasattr(self, 'use_lite_client') and self.use_lite_client:
            if self.nodes_collection_name in self.client.list_collections():
                self.client.drop_collection(self.nodes_collection_name)
            if self.relationships_collection_name in self.client.list_collections():
                self.client.drop_collection(self.relationships_collection_name)
            self._create_collections_lite()
        else:
            if utility.has_collection(self.nodes_collection_name):
                utility.drop_collection(self.nodes_collection_name)
            if utility.has_collection(self.relationships_collection_name):
                utility.drop_collection(self.relationships_collection_name)
            self._create_collections_standard()
        
        self._node_cache.clear()

