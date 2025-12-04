"""
向量数据库实现数据挖掘查询示例
展示如何使用向量数据库实现类似 Neo4j 的数据挖掘功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# # from implementations.chroma_impl import ChromaGraphDB  # 可选：使用 ChromaDB
# from implementations.milvus_impl import MilvusGraphDB  # 可选：使用 Milvus
from implementations.qdrant_impl import QdrantGraphDB  # 使用 Qdrant
# from implementations.orientdb_impl import OrientDBGraphDB  # 可选：使用 OrientDB  # 可选：使用 ChromaDB
from implementations.milvus_impl import MilvusGraphDB
from models.graph_models import Node, Relationship, GraphData
from demo import create_sample_data
import json
from collections import defaultdict

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_table(header, rows, max_rows=15):
    """打印表格"""
    if not rows:
        print("(无结果)")
        return
    
    # 打印表头
    header_str = " | ".join([f"{h:25}" for h in header])
    print(header_str)
    print("-" * len(header_str))
    
    # 打印数据
    for row in rows[:max_rows]:
        row_str = " | ".join([f"{str(cell)[:25]:25}" for cell in row])
        print(row_str)
    
    if len(rows) > max_rows:
        print(f"... 还有 {len(rows) - max_rows} 条记录")

def run_mining_query(db, graph_data: GraphData, query_func, description):
    """运行挖掘查询并显示结果"""
    print(f"\n【{description}】")
    print("-" * 80)
    try:
        results = query_func(db, graph_data)
        return results
    except Exception as e:
        print(f"❌ 查询出错: {e}")
        return None

def main():
    print_section("向量数据库数据挖掘分析")
    
    # 初始化数据库
    print("\n正在初始化 Qdrant...")
    # 使用 Qdrant 本地内存模式（无需启动服务）
    db = QdrantGraphDB(use_local=True)
    
    # 创建并加载数据
    print("正在创建示例数据...")
    graph_data = create_sample_data()
    
    # 添加节点
    print("正在添加节点...")
    for node in graph_data.nodes:
        db.add_node(node)
    
    # 添加关系
    print("正在添加关系...")
    for rel in graph_data.relationships:
        source_node = graph_data.get_node(rel.source_id)
        target_node = graph_data.get_node(rel.target_id)
        if source_node and target_node:
            db.add_relationship(rel, source_node, target_node)
    
    print("✓ 数据加载完成\n")
    
    # 1. 演员合作网络分析
    print_section("1. 演员合作网络分析")
    
    def query_actor_collaborations(db, graph_data):
        """查询演员合作频率"""
        # 统计演员之间的合作次数
        collaborations = defaultdict(int)
        collaboration_movies = defaultdict(list)
        
        # 获取所有 ACTED_IN 关系
        acted_in_rels = graph_data.get_relationships_by_type("ACTED_IN")
        
        # 按电影分组
        movies_actors = defaultdict(list)
        for rel in acted_in_rels:
            movies_actors[rel.target_id].append(rel.source_id)
        
        # 计算每对演员的合作次数
        for movie_id, actor_ids in movies_actors.items():
            movie = graph_data.get_node(movie_id)
            movie_title = movie.properties.get('title', movie_id) if movie else movie_id
            
            for i, actor1_id in enumerate(actor_ids):
                for actor2_id in actor_ids[i+1:]:
                    actor1 = graph_data.get_node(actor1_id)
                    actor2 = graph_data.get_node(actor2_id)
                    
                    if actor1 and actor2:
                        pair = tuple(sorted([actor1.properties.get('name', actor1_id), 
                                             actor2.properties.get('name', actor2_id)]))
                        collaborations[pair] += 1
                        if movie_title not in collaboration_movies[pair]:
                            collaboration_movies[pair].append(movie_title)
        
        # 格式化结果
        rows = []
        for (actor1, actor2), count in sorted(collaborations.items(), key=lambda x: x[1], reverse=True):
            movies = ", ".join(collaboration_movies[(actor1, actor2)])
            rows.append([actor1, actor2, str(count), movies])
        
        print_table(["演员1", "演员2", "合作次数", "合作电影"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_actor_collaborations, "演员合作频率排名")
    
    # 2. 电影推荐系统
    print_section("2. 电影推荐系统")
    
    def query_movie_recommendations(db, graph_data):
        """基于共同参演演员推荐电影"""
        # 找到 Tom Hanks
        tom_nodes = db.find_similar_nodes("Tom Hanks", node_type="Person", top_k=1)
        if not tom_nodes:
            print("未找到 Tom Hanks")
            return []
        
        tom_id = tom_nodes[0]["id"]
        
        # 找到 Tom Hanks 参演的电影
        tom_movies = []
        tom_relationships = db.find_relationships(source_id=tom_id, rel_type="ACTED_IN")
        for rel in tom_relationships:
            movie = db.get_node(rel["target_id"])
            if movie:
                tom_movies.append(movie["id"])
        
        # 找到与 Tom Hanks 共同参演的演员
        co_actors = defaultdict(set)
        for movie_id in tom_movies:
            movie_rels = db.find_relationships(target_id=movie_id, rel_type="ACTED_IN")
            for rel in movie_rels:
                if rel["source_id"] != tom_id:
                    co_actors[rel["source_id"]].add(movie_id)
        
        # 找到这些演员参演的其他电影
        recommendations = defaultdict(int)
        for co_actor_id, common_movies in co_actors.items():
            co_actor_rels = db.find_relationships(source_id=co_actor_id, rel_type="ACTED_IN")
            for rel in co_actor_rels:
                if rel["target_id"] not in tom_movies:
                    recommendations[rel["target_id"]] += 1
        
        # 格式化结果
        rows = []
        for movie_id, count in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]:
            movie = db.get_node(movie_id)
            if movie:
                rows.append([movie.properties.get('title', movie_id), str(count)])
        
        print_table(["推荐电影", "共同演员数"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_movie_recommendations, 
                    "基于 Tom Hanks 的电影推荐（通过共同演员）")
    
    # 3. 路径分析
    print_section("3. 路径分析")
    
    def query_paths(db, graph_data):
        """查找两个演员之间的路径"""
        tom_nodes = db.find_similar_nodes("Tom Hanks", node_type="Person", top_k=1)
        keanu_nodes = db.find_similar_nodes("Keanu Reeves", node_type="Person", top_k=1)
        
        if not tom_nodes or not keanu_nodes:
            print("未找到指定演员")
            return []
        
        tom_id = tom_nodes[0]["id"]
        keanu_id = keanu_nodes[0]["id"]
        
        paths = db.find_path(tom_id, keanu_id, max_hops=2)
        
        rows = []
        for path in paths[:5]:
            path_names = []
            for node_id in path:
                node = db.get_node(node_id)
                if node:
                    name = node['properties'].get('name') or node['properties'].get('title') or node_id
                    path_names.append(name)
            
            rows.append([str(len(path) - 1), " -> ".join(path_names)])
        
        print_table(["路径长度", "路径节点"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_paths, "Tom Hanks 和 Keanu Reeves 之间的路径")
    
    # 4. 影响力分析
    print_section("4. 影响力分析")
    
    def query_degree_centrality(db, graph_data):
        """度中心性：连接最多的节点"""
        # 统计每个节点的连接数
        node_degrees = defaultdict(int)
        
        for rel in graph_data.relationships:
            node_degrees[rel.source_id] += 1
            node_degrees[rel.target_id] += 1
        
        rows = []
        for node_id, degree in sorted(node_degrees.items(), key=lambda x: x[1], reverse=True):
            node = db.get_node(node_id)
            if node:
                name = node['properties'].get('name') or node['properties'].get('title') or node_id
                rows.append([name, node['type'], str(degree)])
        
        print_table(["人物", "类型", "连接数"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_degree_centrality, "连接数排名（影响力分析）")
    
    def query_bridge_analysis(db, graph_data):
        """桥梁人物分析"""
        # 统计每个演员通过电影连接的不同演员数
        bridge_scores = defaultdict(set)
        
        for rel in graph_data.get_relationships_by_type("ACTED_IN"):
            movie_id = rel.target_id
            actor_id = rel.source_id
            
            # 找到同一部电影的其他演员
            movie_rels = db.find_relationships(target_id=movie_id, rel_type="ACTED_IN")
            for other_rel in movie_rels:
                if other_rel["source_id"] != actor_id:
                    bridge_scores[actor_id].add(other_rel["source_id"])
        
        rows = []
        for actor_id, connected_actors in sorted(bridge_scores.items(), 
                                                key=lambda x: len(x[1]), reverse=True):
            actor = db.get_node(actor_id)
            if actor:
                rows.append([actor['properties'].get('name', actor_id), 
                           str(len(connected_actors))])
        
        print_table(["桥梁人物", "连接的不同演员数"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_bridge_analysis, "桥梁人物分析（连接不同演员）")
    
    # 5. 社区发现
    print_section("5. 社区发现")
    
    def query_director_communities(db, graph_data):
        """识别导演的电影系列"""
        director_movies = defaultdict(list)
        
        for rel in graph_data.get_relationships_by_type("DIRECTED"):
            director = db.get_node(rel.source_id)
            movie = db.get_node(rel.target_id)
            
            if director and movie:
                director_name = director['properties'].get('name', rel.source_id)
                movie_title = movie['properties'].get('title', rel.target_id)
                director_movies[director_name].append(movie_title)
        
        rows = []
        for director, movies in sorted(director_movies.items(), 
                                       key=lambda x: len(x[1]), reverse=True):
            if len(movies) > 1:
                rows.append([director, ", ".join(movies), str(len(movies))])
        
        print_table(["导演", "电影列表", "电影数量"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_director_communities, "导演的电影系列（社区）")
    
    def query_actor_communities(db, graph_data):
        """识别紧密合作的演员群体"""
        # 使用之前的合作统计
        collaborations = defaultdict(int)
        
        movies_actors = defaultdict(list)
        for rel in graph_data.get_relationships_by_type("ACTED_IN"):
            movies_actors[rel.target_id].append(rel.source_id)
        
        for movie_id, actor_ids in movies_actors.items():
            for i, actor1_id in enumerate(actor_ids):
                for actor2_id in actor_ids[i+1:]:
                    actor1 = graph_data.get_node(actor1_id)
                    actor2 = graph_data.get_node(actor2_id)
                    
                    if actor1 and actor2:
                        pair = tuple(sorted([actor1.properties.get('name', actor1_id), 
                                             actor2.properties.get('name', actor2_id)]))
                        collaborations[pair] += 1
        
        rows = []
        for (actor1, actor2), count in sorted(collaborations.items(), 
                                            key=lambda x: x[1], reverse=True):
            if count >= 2:
                rows.append([actor1, actor2, str(count)])
        
        print_table(["演员1", "演员2", "合作次数"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_actor_communities, "紧密合作的演员群体")
    
    # 6. 时间序列分析
    print_section("6. 时间序列分析")
    
    def query_career_timeline(db, graph_data):
        """演员职业生涯轨迹"""
        actor_movies = defaultdict(list)
        
        for rel in graph_data.get_relationships_by_type("ACTED_IN"):
            actor = db.get_node(rel.source_id)
            movie = db.get_node(rel.target_id)
            
            if actor and movie and actor['type'] == 'Person':
                actor_name = actor['properties'].get('name', rel.source_id)
                released = movie['properties'].get('released')
                if released:
                    actor_movies[actor_name].append(released)
        
        rows = []
        for actor_name, years in actor_movies.items():
            if years:
                min_year = min(years)
                max_year = max(years)
                count = len(years)
                rows.append([actor_name, str(min_year), str(max_year), str(count)])
        
        rows.sort(key=lambda x: int(x[1]))  # 按首部电影年份排序
        print_table(["演员", "首部电影年份", "最新电影年份", "参演电影数"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_career_timeline, "演员职业生涯时间线")
    
    # 7. 属性挖掘
    print_section("7. 属性挖掘")
    
    def query_year_distribution(db, graph_data):
        """电影年份分布"""
        year_count = defaultdict(int)
        
        for movie in graph_data.get_nodes_by_type("Movie"):
            released = movie.properties.get('released')
            if released:
                year_count[released] += 1
        
        rows = []
        for year in sorted(year_count.keys()):
            rows.append([str(year), str(year_count[year])])
        
        print_table(["年份", "电影数量"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_year_distribution, "电影年份分布")
    
    # 8. 关系强度分析
    print_section("8. 关系强度分析")
    
    def query_director_actor_collaboration(db, graph_data):
        """导演-演员合作强度"""
        collaborations = defaultdict(list)
        
        # 找到所有导演-电影关系
        for dir_rel in graph_data.get_relationships_by_type("DIRECTED"):
            director = db.get_node(dir_rel.source_id)
            movie = db.get_node(dir_rel.target_id)
            
            if not director or not movie:
                continue
            
            if director and movie:
                # 找到这部电影的演员
                movie_actor_rels = db.find_relationships(target_id=dir_rel.target_id, 
                                                         rel_type="ACTED_IN")
                for actor_rel in movie_actor_rels:
                    actor = db.get_node(actor_rel.source_id)
                    if actor:
                        key = (director['properties'].get('name', dir_rel.source_id),
                              actor['properties'].get('name', actor_rel.source_id))
                        collaborations[key].append(movie['properties'].get('title', dir_rel.target_id))
        
        rows = []
        for (director, actor), movies in sorted(collaborations.items(), 
                                               key=lambda x: len(x[1]), reverse=True):
            rows.append([director, actor, str(len(movies)), ", ".join(movies)])
        
        print_table(["导演", "演员", "合作次数", "合作电影"], rows)
        return rows
    
    run_mining_query(db, graph_data, query_director_actor_collaboration, "导演-演员合作强度")
    
    print_section("挖掘分析完成")
    print("\n💡 提示：")
    print("  - 向量数据库通过相似度搜索和元数据过滤实现图查询")
    print("  - 可以结合语义搜索进行更智能的查询")
    print("  - 适合推荐系统和相似度匹配场景")

if __name__ == "__main__":
    main()

