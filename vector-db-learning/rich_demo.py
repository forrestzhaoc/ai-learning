"""
丰富数据演示 - 展示向量数据库的优势
"""

from implementations.qdrant_impl import QdrantGraphDB
from rich_data_generator import create_rich_data
import json

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    print_section("向量数据库优势演示 - 丰富数据集")
    
    # 初始化数据库
    print("\n正在初始化 Qdrant...")
    db = QdrantGraphDB(use_local=True)
    
    # 创建丰富数据
    print("\n正在创建丰富的数据集..."
    graph_data = create_rich_data()
    
    # 添加数据
    print("\n正在添加数据到向量数据库...")
    for node in graph_data.nodes:
        db.add_node(node)
    for rel in graph_data.relationships:
        source_node = graph_data.get_node(rel.source_id)
        target_node = graph_data.get_node(rel.target_id)
        if source_node and target_node:
            db.add_relationship(rel, source_node, target_node)
    
    print(f"✓ 已添加 {len(graph_data.nodes)} 个节点")
    print(f"✓ 已添加 {len(graph_data.relationships)} 个关系")
    
    # ========== 演示1: 语义搜索优势 ==========
    print_section("演示1: 语义搜索 - 理解查询意图")
    
    print("\n【查询：'科幻动作片'】")
    print("向量数据库可以理解语义，找到相关电影，即使没有精确匹配")
    results = db.find_similar_nodes("科幻动作片", node_type="Movie", top_k=5)
    for movie in results:
        title = movie['properties'].get('title', 'Unknown')
        genre = movie['properties'].get('genre', '')
        similarity = movie.get('similarity', 0)
        print(f"  - {title} ({genre}) - 相似度: {similarity:.4f}")
    
    print("\n【查询：'获得奥斯卡奖的演员'】")
    results = db.find_similar_nodes("获得奥斯卡奖的演员", node_type="Person", top_k=5)
    for person in results:
        name = person['properties'].get('name', 'Unknown')
        awards = person['properties'].get('awards', '')
        similarity = person.get('similarity', 0)
        print(f"  - {name} ({awards}) - 相似度: {similarity:.4f}")
    
    # ========== 演示2: 多属性查询 ==========
    print_section("演示2: 多属性组合查询")
    
    print("\n【查找：高评分科幻电影】")
    # 先通过语义找到科幻电影
    sci_fi_movies = db.find_similar_nodes("sci-fi science fiction", node_type="Movie", top_k=10)
    print("  找到的科幻电影:")
    for movie in sci_fi_movies:
        title = movie['properties'].get('title', 'Unknown')
        rating_str = movie['properties'].get('rating', '0')
        released_str = movie['properties'].get('released', '0')
        try:
            rating = float(rating_str) if rating_str else 0
            released = int(released_str) if released_str else 0
            if rating and rating >= 8.5:
                print(f"    ⭐ {title} (评分: {rating}, 年份: {released})")
        except (ValueError, TypeError):
            pass
    
    # ========== 演示3: 复杂关系查询 ==========
    print_section("演示3: 复杂关系网络分析")
    
    print("\n【查找：Christopher Nolan 的电影宇宙】")
    nolan = db.find_similar_nodes("Christopher Nolan", node_type="Person", top_k=1)
    if nolan:
        nolan_id = nolan[0]['id']
        # 查找他导演的电影
        directed = db.find_relationships(source_id=nolan_id, rel_type="DIRECTED")
        # 查找他编剧的电影
        wrote = db.find_relationships(source_id=nolan_id, rel_type="WROTE")
        
        print(f"  {nolan[0]['properties'].get('name')} 的作品:")
        all_movies = set()
        for rel in directed:
            movie = db.get_node(rel['target_id'])
            if movie:
                all_movies.add(movie['properties'].get('title'))
        for rel in wrote:
            movie = db.get_node(rel['target_id'])
            if movie:
                all_movies.add(movie['properties'].get('title'))
        
        for movie_title in sorted(all_movies):
            print(f"    - {movie_title}")
    
    # ========== 演示4: 推荐系统 ==========
    print_section("演示4: 智能推荐系统")
    
    print("\n【基于用户喜欢 'The Matrix' 推荐相似电影】")
    matrix = db.find_similar_nodes("The Matrix", node_type="Movie", top_k=1)[0]
    recommendations = db.find_similar_nodes(
        matrix['properties'].get('title', 'The Matrix'),
        node_type="Movie",
        top_k=6
    )
    
    print("  推荐电影（基于语义相似度）:")
    for i, movie in enumerate(recommendations[1:6], 1):  # 跳过自己
        title = movie['properties'].get('title', 'Unknown')
        genre = movie['properties'].get('genre', '')
        similarity = movie.get('similarity', 0)
        print(f"    {i}. {title} ({genre}) - 相似度: {similarity:.4f}")
    
    # ========== 演示5: 跨维度查询 ==========
    print_section("演示5: 跨维度语义查询")
    
    print("\n【查询：'90年代的经典电影'】")
    results = db.find_similar_nodes("90年代的经典电影", node_type="Movie", top_k=5)
    for movie in results:
        title = movie['properties'].get('title', 'Unknown')
        released_str = movie['properties'].get('released', '0')
        rating_str = movie['properties'].get('rating', '0')
        try:
            released = int(released_str) if released_str else 0
            rating = float(rating_str) if rating_str else 0
            if 1990 <= released < 2000:
                print(f"  - {title} ({released}, 评分: {rating})")
        except (ValueError, TypeError):
            pass
    
    print("\n【查询：'英国演员'】")
    results = db.find_similar_nodes("British actor", node_type="Person", top_k=5)
    for person in results:
        name = person['properties'].get('name', 'Unknown')
        nationality = person['properties'].get('nationality', '')
        print(f"  - {name} ({nationality})")
    
    # ========== 演示6: 关系强度分析 ==========
    print_section("演示6: 关系强度和多跳查询")
    
    print("\n【查找：与 Tom Hanks 合作过的导演】")
    tom = db.find_similar_nodes("Tom Hanks", node_type="Person", top_k=1)[0]
    tom_id = tom['id']
    
    # 找到 Tom 参演的电影
    tom_movies = db.find_relationships(source_id=tom_id, rel_type="ACTED_IN")
    directors = set()
    for rel in tom_movies:
        movie_id = rel['target_id']
        # 找到这些电影的导演
        movie_directors = db.find_relationships(target_id=movie_id, rel_type="DIRECTED")
        for dir_rel in movie_directors:
            director = db.get_node(dir_rel['source_id'])
            if director:
                directors.add(director['properties'].get('name'))
    
    print(f"  {tom['properties'].get('name')} 合作过的导演:")
    for director in sorted(directors):
        print(f"    - {director}")
    
    # ========== 演示7: 属性挖掘 ==========
    print_section("演示7: 基于属性的智能分析")
    
    print("\n【查找：高票房电影】")
    all_movies = db.find_similar_nodes("movie film", node_type="Movie", top_k=20)
    high_box_office = []
    for movie in all_movies:
        box_office_str = movie['properties'].get('box_office', '0')
        try:
            box_office = int(box_office_str) if box_office_str else 0
            if box_office and box_office > 500000000:  # 5亿以上
                high_box_office.append((movie['properties'].get('title'), box_office))
        except (ValueError, TypeError):
            pass
    
    high_box_office.sort(key=lambda x: x[1], reverse=True)
    print("  高票房电影（5亿美元以上）:")
    for title, revenue in high_box_office[:5]:
        print(f"    - {title}: ${revenue/1000000:.0f}M")
    
    print("\n【查找：获奖导演】")
    directors = db.find_similar_nodes("director filmmaker", node_type="Person", top_k=15)
    award_winners = []
    for director in directors:
        awards = director['properties'].get('awards', '')
        if 'Oscar' in awards or 'Winner' in awards:
            award_winners.append(director['properties'].get('name'))
    
    print("  获奖导演:")
    for name in sorted(set(award_winners)):
        print(f"    - {name}")
    
    print_section("演示完成")
    print("\n💡 向量数据库的优势总结:")
    print("  ✓ 语义理解：可以理解查询意图，不需要精确匹配")
    print("  ✓ 多维度搜索：可以同时考虑多个属性进行相似度匹配")
    print("  ✓ 智能推荐：基于语义相似度进行推荐")
    print("  ✓ 复杂关系：可以轻松处理多跳关系和复杂网络")
    print("  ✓ 属性挖掘：可以基于向量相似度发现隐藏的模式")

if __name__ == "__main__":
    main()

