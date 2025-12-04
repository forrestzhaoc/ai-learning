"""
向量数据库实现图数据库场景 - 基础演示
展示如何使用向量数据库存储和查询图数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from implementations.chroma_impl import ChromaGraphDB  # 可选：使用 ChromaDB
# from implementations.milvus_impl import MilvusGraphDB  # 可选：使用 Milvus
from implementations.qdrant_impl import QdrantGraphDB  # 使用 Qdrant
from models.graph_models import Node, Relationship, GraphData
import json

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_node(node_dict):
    """打印节点信息"""
    print(f"  ID: {node_dict['id']}")
    print(f"  类型: {node_dict['type']}")
    print(f"  属性: {json.dumps(node_dict['properties'], ensure_ascii=False, indent=4)}")
    if 'similarity' in node_dict and node_dict['similarity']:
        print(f"  相似度: {node_dict['similarity']:.4f}")

def print_relationship(rel_dict):
    """打印关系信息"""
    print(f"  ID: {rel_dict['id']}")
    print(f"  类型: {rel_dict['type']}")
    print(f"  源节点: {rel_dict['source_id']}")
    print(f"  目标节点: {rel_dict['target_id']}")
    if rel_dict['properties']:
        print(f"  属性: {json.dumps(rel_dict['properties'], ensure_ascii=False, indent=4)}")

def create_sample_data():
    """创建示例数据（与 Neo4j demo 类似）"""
    graph_data = GraphData()
    
    # 创建人物节点
    people = [
        Node("p1", "Person", {"name": "Tom Hanks", "born": 1956}),
        Node("p2", "Person", {"name": "Keanu Reeves", "born": 1964}),
        Node("p3", "Person", {"name": "Carrie-Anne Moss", "born": 1967}),
        Node("p4", "Person", {"name": "Laurence Fishburne", "born": 1961}),
        Node("p5", "Person", {"name": "Hugo Weaving", "born": 1960}),
        Node("p6", "Person", {"name": "Lana Wachowski", "born": 1965}),
        Node("p7", "Person", {"name": "Lilly Wachowski", "born": 1967}),
        Node("p8", "Person", {"name": "Robert Zemeckis", "born": 1952}),
        Node("p9", "Person", {"name": "Robin Wright", "born": 1966}),
        Node("p10", "Person", {"name": "Gary Sinise", "born": 1955}),
    ]
    
    # 创建电影节点
    movies = [
        Node("m1", "Movie", {"title": "The Matrix", "released": 1999, "tagline": "Welcome to the Real World"}),
        Node("m2", "Movie", {"title": "The Matrix Reloaded", "released": 2003, "tagline": "Free your mind"}),
        Node("m3", "Movie", {"title": "The Matrix Revolutions", "released": 2003, "tagline": "Everything that has a beginning has an end"}),
        Node("m4", "Movie", {"title": "Forrest Gump", "released": 1994, "tagline": "Life is like a box of chocolates"}),
        Node("m5", "Movie", {"title": "The Green Mile", "released": 1999, "tagline": "Walk a mile you'll never forget"}),
    ]
    
    # 添加节点
    for person in people:
        graph_data.add_node(person)
    for movie in movies:
        graph_data.add_node(movie)
    
    # 创建关系
    relationships = [
        # The Matrix 系列
        Relationship("r1", "p2", "m1", "ACTED_IN", {"roles": ["Neo"]}),
        Relationship("r2", "p3", "m1", "ACTED_IN", {"roles": ["Trinity"]}),
        Relationship("r3", "p4", "m1", "ACTED_IN", {"roles": ["Morpheus"]}),
        Relationship("r4", "p5", "m1", "ACTED_IN", {"roles": ["Agent Smith"]}),
        Relationship("r5", "p6", "m1", "DIRECTED", {}),
        Relationship("r6", "p7", "m1", "DIRECTED", {}),
        
        Relationship("r7", "p2", "m2", "ACTED_IN", {"roles": ["Neo"]}),
        Relationship("r8", "p3", "m2", "ACTED_IN", {"roles": ["Trinity"]}),
        Relationship("r9", "p4", "m2", "ACTED_IN", {"roles": ["Morpheus"]}),
        Relationship("r10", "p6", "m2", "DIRECTED", {}),
        Relationship("r11", "p7", "m2", "DIRECTED", {}),
        
        Relationship("r12", "p2", "m3", "ACTED_IN", {"roles": ["Neo"]}),
        Relationship("r13", "p3", "m3", "ACTED_IN", {"roles": ["Trinity"]}),
        Relationship("r14", "p4", "m3", "ACTED_IN", {"roles": ["Morpheus"]}),
        Relationship("r15", "p6", "m3", "DIRECTED", {}),
        Relationship("r16", "p7", "m3", "DIRECTED", {}),
        
        # Forrest Gump
        Relationship("r17", "p1", "m4", "ACTED_IN", {"roles": ["Forrest Gump"]}),
        Relationship("r18", "p9", "m4", "ACTED_IN", {"roles": ["Jenny Curran"]}),
        Relationship("r19", "p10", "m4", "ACTED_IN", {"roles": ["Lieutenant Dan Taylor"]}),
        Relationship("r20", "p8", "m4", "DIRECTED", {}),
        
        # The Green Mile
        Relationship("r21", "p1", "m5", "ACTED_IN", {"roles": ["Paul Edgecomb"]}),
    ]
    
    # 添加关系
    for rel in relationships:
        source_node = graph_data.get_node(rel.source_id)
        target_node = graph_data.get_node(rel.target_id)
        if source_node and target_node:
            graph_data.add_relationship(rel)
    
    return graph_data

def main():
    print_section("向量数据库实现图数据库场景 - 基础演示")
    
    # 初始化数据库
    print("\n正在初始化 Qdrant...")
    # 使用 Qdrant 本地内存模式（无需启动服务）
    db = QdrantGraphDB(use_local=True)
    # 或使用 Qdrant 服务模式：
    # db = QdrantGraphDB(host="localhost", port=6333, use_local=False)
    
    # 清空现有数据（可选）
    # db.clear()
    
    # 创建示例数据
    print_section("1. 创建示例数据")
    graph_data = create_sample_data()
    
    # 添加节点到数据库
    print("\n正在添加节点到向量数据库...")
    for node in graph_data.nodes:
        db.add_node(node)
    print(f"✓ 已添加 {len(graph_data.nodes)} 个节点")
    
    # 添加关系到数据库
    print("\n正在添加关系到向量数据库...")
    for rel in graph_data.relationships:
        source_node = graph_data.get_node(rel.source_id)
        target_node = graph_data.get_node(rel.target_id)
        if source_node and target_node:
            db.add_relationship(rel, source_node, target_node)
    print(f"✓ 已添加 {len(graph_data.relationships)} 个关系")
    
    # 2. 节点查询
    print_section("2. 节点查询 - 相似度搜索")
    
    print("\n【查找与 'Tom Hanks' 相似的节点】")
    similar_nodes = db.find_similar_nodes("Tom Hanks", top_k=5)
    for node in similar_nodes:
        print_node(node)
        print()
    
    print("\n【查找与 'The Matrix' 相似的节点】")
    similar_nodes = db.find_similar_nodes("The Matrix", top_k=5)
    for node in similar_nodes:
        print_node(node)
        print()
    
    # 3. 关系查询
    print_section("3. 关系查询")
    
    print("\n【查找 Tom Hanks 参演的电影】")
    tom_hanks_node = db.find_similar_nodes("Tom Hanks", node_type="Person", top_k=1)
    if tom_hanks_node:
        tom_id = tom_hanks_node[0]["id"]
        relationships = db.find_relationships(source_id=tom_id, rel_type="ACTED_IN")
        for rel in relationships:
            target_node = db.get_node(rel["target_id"])
            if target_node:
                print(f"  {tom_hanks_node[0]['properties'].get('name')} 参演了 {target_node['properties'].get('title')}")
    
    print("\n【查找 The Matrix 的演员】")
    matrix_node = db.find_similar_nodes("The Matrix", node_type="Movie", top_k=1)
    if matrix_node:
        matrix_id = matrix_node[0]["id"]
        relationships = db.find_relationships(target_id=matrix_id, rel_type="ACTED_IN")
        for rel in relationships:
            source_node = db.get_node(rel["source_id"])
            if source_node:
                print(f"  {source_node['properties'].get('name')} 参演了 {matrix_node[0]['properties'].get('title')}")
    
    # 4. 路径查找
    print_section("4. 路径查找")
    
    print("\n【查找 Tom Hanks 和 Keanu Reeves 之间的路径】")
    tom_node = db.find_similar_nodes("Tom Hanks", node_type="Person", top_k=1)
    keanu_node = db.find_similar_nodes("Keanu Reeves", node_type="Person", top_k=1)
    
    if tom_node and keanu_node:
        tom_id = tom_node[0]["id"]
        keanu_id = keanu_node[0]["id"]
        paths = db.find_path(tom_id, keanu_id, max_hops=3)
        
        if paths:
            for i, path in enumerate(paths[:3], 1):  # 最多显示3条路径
                print(f"\n  路径 {i}:")
                path_names = []
                for node_id in path:
                    node = db.get_node(node_id)
                    if node:
                        name = node['properties'].get('name') or node['properties'].get('title') or node_id
                        path_names.append(name)
                print(f"    {' -> '.join(path_names)}")
        else:
            print("  (未找到路径)")
    
    # 5. 推荐系统
    print_section("5. 推荐系统 - 基于相似度")
    
    print("\n【基于 'The Matrix' 推荐相似电影】")
    matrix_node = db.find_similar_nodes("The Matrix", node_type="Movie", top_k=1)
    if matrix_node:
        recommendations = db.find_similar_nodes(
            matrix_node[0]['properties'].get('title', 'The Matrix'),
            node_type="Movie",
            top_k=5
        )
        for rec in recommendations[1:]:  # 跳过自己
            print(f"  - {rec['properties'].get('title')} (相似度: {rec.get('similarity', 0):.4f})")
    
    print_section("演示完成")
    print("\n💡 提示：")
    print("  - 向量数据库通过语义相似度搜索实现图查询")
    print("  - 可以结合 LLM 进行自然语言查询")
    print("  - 适合推荐系统和相似度匹配场景")

if __name__ == "__main__":
    main()

