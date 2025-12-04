"""
OrientDB 图数据库实现
使用 OrientDB 存储图数据，向量嵌入作为节点属性
"""

try:
    from pyorient import OrientDB
    ORIENTDB_AVAILABLE = True
except ImportError:
    ORIENTDB_AVAILABLE = False
    print("⚠ pyorient 未安装，请运行: pip install pyorient")

from typing import List, Dict, Any, Optional
import numpy as np
import json
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

# 向量维度
EMBEDDING_DIM = 384

class OrientDBGraphDB:
    """使用 OrientDB 实现的图数据库接口"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 2424,
                 database: str = "graph_db",
                 username: str = "admin",
                 password: str = "admin",
                 use_embedded: bool = False):
        """
        初始化 OrientDB 图数据库
        
        Args:
            host: OrientDB 服务器地址
            port: OrientDB 服务器端口
            database: 数据库名称
            username: 用户名
            password: 密码
            use_embedded: 是否使用嵌入式模式（需要 OrientDB 本地安装）
        """
        if not ORIENTDB_AVAILABLE:
            raise ImportError("pyorient 未安装，请运行: pip install pyorient")
        
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        
        # 连接 OrientDB
        try:
            self.client = OrientDB(host, port)
            self.client.connect(username, password)
            print(f"✓ 已连接到 OrientDB 服务器 ({host}:{port})")
        except Exception as e:
            raise Exception(f"无法连接到 OrientDB 服务器 ({host}:{port}): {e}\n"
                          f"请确保 OrientDB 服务正在运行。启动方法：\n"
                          f"  docker run -d --name orientdb -p 2424:2424 -p 2480:2480 -e ORIENTDB_ROOT_PASSWORD=root orientdb:latest")
        
        # 检查数据库是否存在，如果不存在则创建
        try:
            self.client.db_open(database, username, password)
            print(f"✓ 已连接到 OrientDB 数据库: {database}")
        except Exception as e:
            # 数据库不存在，创建它
            try:
                self.client.db_create(database, OrientDB.DB_TYPE_GRAPH, OrientDB.STORAGE_TYPE_MEMORY)
                self.client.db_open(database, username, password)
                print(f"✓ 已创建并连接到 OrientDB 数据库: {database}")
            except Exception as e2:
                print(f"⚠ 创建数据库失败: {e2}")
                print("尝试使用已存在的数据库...")
                try:
                    self.client.db_open(database, username, password)
                    print(f"✓ 已连接到 OrientDB 数据库: {database}")
                except Exception as e3:
                    raise Exception(f"无法连接或创建数据库 '{database}': {e3}\n"
                                  f"请检查数据库名称和权限设置")
        
        # 创建类和索引
        self._create_schema()
        
        # 节点元数据缓存
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        
        # 加载嵌入模型
        self.embedding_model = get_embedding_model()
    
    def _create_schema(self):
        """创建数据库模式（类和索引）"""
        # 创建节点类
        try:
            self.client.command("CREATE CLASS Node IF NOT EXISTS EXTENDS V")
            print("✓ 节点类已创建/存在")
        except Exception as e:
            print(f"创建节点类时出错: {e}")
        
        # 创建关系类
        try:
            self.client.command("CREATE CLASS Edge IF NOT EXISTS EXTENDS E")
            print("✓ 关系类已创建/存在")
        except Exception as e:
            print(f"创建关系类时出错: {e}")
        
        # 为节点创建属性
        try:
            self.client.command("CREATE PROPERTY Node.id IF NOT EXISTS STRING")
            self.client.command("CREATE PROPERTY Node.node_type IF NOT EXISTS STRING")
            self.client.command("CREATE PROPERTY Node.properties IF NOT EXISTS EMBEDDEDMAP")
            self.client.command("CREATE PROPERTY Node.embedding IF NOT EXISTS EMBEDDEDLIST")
            
            # 创建索引
            self.client.command("CREATE INDEX Node.id IF NOT EXISTS UNIQUE")
            print("✓ 节点属性和索引已创建")
        except Exception as e:
            print(f"创建节点属性时出错: {e}")
        
        # 为关系创建属性
        try:
            self.client.command("CREATE PROPERTY Edge.id IF NOT EXISTS STRING")
            self.client.command("CREATE PROPERTY Edge.rel_type IF NOT EXISTS STRING")
            self.client.command("CREATE PROPERTY Edge.properties IF NOT EXISTS EMBEDDEDMAP")
            self.client.command("CREATE PROPERTY Edge.embedding IF NOT EXISTS EMBEDDEDLIST")
            
            # 创建索引
            self.client.command("CREATE INDEX Edge.id IF NOT EXISTS UNIQUE")
            print("✓ 关系属性和索引已创建")
        except Exception as e:
            print(f"创建关系属性时出错: {e}")
    
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
        
        # 检查节点是否已存在
        try:
            existing = self.client.query(f"SELECT FROM Node WHERE id = '{node.id}'")
            if existing and len(existing) > 0:
                # 更新现有节点
                rid = existing[0]._rid
                # OrientDB 可以直接使用字典
                self.client.command(
                    f"UPDATE Node SET node_type = '{node.type}', "
                    f"properties = {node.properties}, "
                    f"embedding = {embedding.tolist()} "
                    f"WHERE @rid = {rid}"
                )
            else:
                # 创建新节点
                self.client.command(
                    f"CREATE VERTEX Node SET "
                    f"id = '{node.id}', "
                    f"node_type = '{node.type}', "
                    f"properties = {node.properties}, "
                    f"embedding = {embedding.tolist()}"
                )
        except Exception as e:
            # 如果上面的方式失败，使用 JSON 字符串
            try:
                props_str = json.dumps(node.properties)
                embedding_list = embedding.tolist()
                self.client.command(
                    f"CREATE VERTEX Node SET "
                    f"id = '{node.id}', "
                    f"node_type = '{node.type}', "
                    f"properties = {props_str}, "
                    f"embedding = {embedding_list}"
                )
            except Exception as e2:
                print(f"添加节点失败: {e2}")
                raise
        
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
        
        # 获取源节点和目标节点的 RID
        source_rid = self.client.query(f"SELECT @rid FROM Node WHERE id = '{source_node.id}'")[0]._rid
        target_rid = self.client.query(f"SELECT @rid FROM Node WHERE id = '{target_node.id}'")[0]._rid
        
        # 检查关系是否已存在
        try:
            existing = self.client.query(
                f"SELECT FROM Edge WHERE id = '{relationship.id}'"
            )
            
            if existing and len(existing) > 0:
                # 更新现有关系
                rid = existing[0]._rid
                self.client.command(
                    f"UPDATE Edge SET rel_type = '{relationship.type}', "
                    f"properties = {relationship.properties}, "
                    f"embedding = {embedding.tolist()} "
                    f"WHERE @rid = {rid}"
                )
            else:
                # 创建新关系
                self.client.command(
                    f"CREATE EDGE Edge FROM {source_rid} TO {target_rid} SET "
                    f"id = '{relationship.id}', "
                    f"rel_type = '{relationship.type}', "
                    f"properties = {relationship.properties}, "
                    f"embedding = {embedding.tolist()}"
                )
        except Exception as e:
            # 如果失败，使用 JSON 字符串
            try:
                props_str = json.dumps(relationship.properties)
                embedding_list = embedding.tolist()
                self.client.command(
                    f"CREATE EDGE Edge FROM {source_rid} TO {target_rid} SET "
                    f"id = '{relationship.id}', "
                    f"rel_type = '{relationship.type}', "
                    f"properties = {props_str}, "
                    f"embedding = {embedding_list}"
                )
            except Exception as e2:
                print(f"添加关系失败: {e2}")
                raise
    
    def find_similar_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找相似节点（使用向量相似度）
        
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
        
        # 构建查询
        where_clause = ""
        if node_type:
            where_clause = f"WHERE node_type = '{node_type}'"
        
        # 获取所有节点（OrientDB 不直接支持向量搜索，需要手动计算相似度）
        query_sql = f"SELECT FROM Node {where_clause}"
        nodes = self.client.query(query_sql)
        
        # 计算相似度并排序
        similarities = []
        for node in nodes:
            try:
                embedding = np.array(node.oRecordData.get('embedding', []))
                if len(embedding) > 0:
                    # 计算余弦相似度
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    similarities.append({
                        'node': node,
                        'similarity': float(similarity)
                    })
            except Exception:
                continue
        
        # 排序并取前 k 个
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 格式化结果
        similar_nodes = []
        for item in similarities[:top_k]:
            node = item['node']
            node_data = node.oRecordData
            similar_nodes.append({
                "id": node_data.get('id'),
                "type": node_data.get('node_type'),
                "properties": node_data.get('properties', {}),
                "similarity": item['similarity']
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
        # 构建查询条件
        conditions = []
        if source_id:
            # 需要先找到源节点的 RID
            source_nodes = self.client.query(f"SELECT @rid FROM Node WHERE id = '{source_id}'")
            if source_nodes:
                source_rid = source_nodes[0]._rid
                conditions.append(f"out = {source_rid}")
        
        if target_id:
            # 需要先找到目标节点的 RID
            target_nodes = self.client.query(f"SELECT @rid FROM Node WHERE id = '{target_id}'")
            if target_nodes:
                target_rid = target_nodes[0]._rid
                conditions.append(f"in = {target_rid}")
        
        if rel_type:
            conditions.append(f"rel_type = '{rel_type}'")
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        # 查询关系
        query_sql = f"SELECT FROM Edge {where_clause} LIMIT {top_k}"
        edges = self.client.query(query_sql)
        
        # 格式化结果
        relationships = []
        for edge in edges:
            edge_data = edge.oRecordData
            # 获取源节点和目标节点 ID
            out_rid = edge._out
            in_rid = edge._in
            
            out_node = self.client.query(f"SELECT id FROM Node WHERE @rid = {out_rid}")
            in_node = self.client.query(f"SELECT id FROM Node WHERE @rid = {in_rid}")
            
            source_id_found = out_node[0].oRecordData.get('id') if out_node else None
            target_id_found = in_node[0].oRecordData.get('id') if in_node else None
            
            relationships.append({
                "id": edge_data.get('id'),
                "source_id": source_id_found,
                "target_id": target_id_found,
                "type": edge_data.get('rel_type'),
                "properties": edge_data.get('properties', {})
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
        
        # 从 OrientDB 查询
        nodes = self.client.query(f"SELECT FROM Node WHERE id = '{node_id}'")
        if nodes:
            node_data = nodes[0].oRecordData
            node_dict = {
                "id": node_data.get('id'),
                "type": node_data.get('node_type'),
                "properties": node_data.get('properties', {})
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
        查找两个节点之间的路径（使用 OrientDB 的图遍历）
        
        Args:
            start_node_id: 起始节点 ID
            end_node_id: 结束节点 ID
            max_hops: 最大跳数
            
        Returns:
            路径列表（每个路径是节点 ID 列表）
        """
        # 获取起始和目标节点的 RID
        start_nodes = self.client.query(f"SELECT @rid FROM Node WHERE id = '{start_node_id}'")
        end_nodes = self.client.query(f"SELECT @rid FROM Node WHERE id = '{end_node_id}'")
        
        if not start_nodes or not end_nodes:
            return []
        
        start_rid = start_nodes[0]._rid
        end_rid = end_nodes[0]._rid
        
        # 使用 OrientDB 的 SHORTESTPATH 函数
        try:
            paths = self.client.query(
                f"SELECT shortestPath({start_rid}, {end_rid}, 'BOTH', {max_hops}).asString() as path "
                f"FROM Node WHERE @rid = {start_rid}"
            )
            
            result_paths = []
            for path_result in paths:
                path_str = path_result.oRecordData.get('path', '')
                if path_str:
                    # 解析路径字符串，提取节点 ID
                    # 这里简化处理，实际需要根据 OrientDB 返回格式解析
                    result_paths.append([start_node_id, end_node_id])
            
            return result_paths
        except Exception:
            # 如果 SHORTESTPATH 不可用，使用简单的遍历
            return self._simple_path_find(start_node_id, end_node_id, max_hops)
    
    def _simple_path_find(self, start_id: str, end_id: str, max_hops: int) -> List[List[str]]:
        """简单的路径查找实现"""
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[str], hops: int):
            if hops > max_hops:
                return
            
            if current_id == end_id:
                paths.append(path.copy())
                return
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            # 查找从当前节点出发的关系
            relationships = self.find_relationships(source_id=current_id)
            for rel in relationships:
                next_id = rel["target_id"]
                if next_id and next_id not in path:
                    path.append(next_id)
                    dfs(next_id, path, hops + 1)
                    path.pop()
            
            visited.remove(current_id)
        
        dfs(start_id, [start_id], 0)
        return paths
    
    def clear(self):
        """清空数据库"""
        try:
            self.client.command("DELETE VERTEX Node")
            self.client.command("DELETE EDGE Edge")
            print("✓ 数据库已清空")
        except Exception as e:
            print(f"清空数据库时出错: {e}")
        
        self._node_cache.clear()
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.db_close()
            self.client.close()

