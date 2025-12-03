"""
Neo4j Demo 可视化图形展示
使用 ASCII 艺术展示图数据库结构
"""

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {title:^76}")
    print("=" * 80 + "\n")

def visualize_graph(driver):
    """可视化图结构"""
    print_header("Neo4j 图数据库结构可视化")
    
    with driver.session() as session:
        # 获取所有电影及其关系
        result = session.run("""
            MATCH (m:Movie)
            OPTIONAL MATCH (m)<-[:ACTED_IN]-(actor:Person)
            OPTIONAL MATCH (m)<-[:DIRECTED]-(director:Person)
            OPTIONAL MATCH (m)<-[:PRODUCED]-(producer:Person)
            OPTIONAL MATCH (m)<-[:REVIEWED]-(reviewer:Person)
            RETURN m.title AS movie,
                   collect(DISTINCT actor.name) AS actors,
                   collect(DISTINCT director.name) AS directors,
                   collect(DISTINCT producer.name) AS producers,
                   collect(DISTINCT reviewer.name) AS reviewers
            ORDER BY movie
        """)
        
        movies = [record for record in result]
        
        print("🎬 电影图结构：\n")
        for movie in movies:
            title = movie['movie']
            actors = [a for a in movie['actors'] if a]
            directors = [d for d in movie['directors'] if d]
            producers = [p for p in movie['producers'] if p]
            reviewers = [r for r in movie['reviewers'] if r]
            
            print(f"┌─ 《{title}》")
            if directors:
                print(f"│  📽️  导演: {', '.join(directors)}")
            if producers:
                print(f"│  🎬 制片: {', '.join(producers)}")
            if actors:
                print(f"│  🎭 演员:")
                for actor in actors:
                    print(f"│     • {actor}")
            if reviewers:
                print(f"│  ⭐ 评论: {', '.join(reviewers)}")
            print("└─\n")
        
        # 演员合作网络
        print("\n" + "-" * 80)
        print("🤝 演员合作网络：\n")
        
        result = session.run("""
            MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
            WHERE p1 <> p2 AND p1.name < p2.name
            RETURN p1.name AS actor1, p2.name AS actor2, 
                   collect(m.title) AS movies, count(m) AS count
            ORDER BY count DESC, actor1
        """)
        
        collaborations = [record for record in result]
        for collab in collaborations:
            movies_list = ', '.join(collab['movies'])
            print(f"  {collab['actor1']:20} ←→ {collab['actor2']:20}  ({collab['count']} 部电影)")
            print(f"    合作电影: {movies_list}\n")

def show_statistics(driver):
    """显示统计信息"""
    print_header("数据库统计信息")
    
    with driver.session() as session:
        # 节点统计
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        
        print("📊 节点统计：")
        total_nodes = 0
        for record in result:
            count = record['count']
            total_nodes += count
            print(f"   {record['label']:15} : {count:3} 个")
        print(f"   {'总计':15} : {total_nodes:3} 个\n")
        
        # 关系统计
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)
        
        print("🔗 关系统计：")
        total_rels = 0
        for record in result:
            count = record['count']
            total_rels += count
            print(f"   {record['type']:15} : {count:3} 个")
        print(f"   {'总计':15} : {total_rels:3} 个\n")

def show_top_queries(driver):
    """显示热门查询结果"""
    print_header("热门查询结果")
    
    with driver.session() as session:
        # 参演电影最多的演员
        print("🏆 参演电影最多的演员：")
        result = session.run("""
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            RETURN p.name AS actor, count(m) AS movie_count
            ORDER BY movie_count DESC
            LIMIT 5
        """)
        
        for i, record in enumerate(result, 1):
            print(f"   {i}. {record['actor']:25} - {record['movie_count']} 部电影")
        
        print()
        
        # 演员最多的电影
        print("🎬 演员最多的电影：")
        result = session.run("""
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            RETURN m.title AS movie, count(p) AS actor_count
            ORDER BY actor_count DESC
            LIMIT 5
        """)
        
        for i, record in enumerate(result, 1):
            print(f"   {i}. {record['movie']:35} - {record['actor_count']} 位演员")
        
        print()

def main():
    """主函数"""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    try:
        # 可视化图结构
        visualize_graph(driver)
        
        # 显示统计信息
        show_statistics(driver)
        
        # 显示热门查询
        show_top_queries(driver)
        
        print_header("Demo 完成！")
        print("\n💡 提示：")
        print("   • 访问 http://localhost:7474 查看交互式图可视化")
        print("   • 运行 'python show_demo.py' 查看详细查询结果")
        print("   • 查看 'cypher_examples.cypher' 学习更多查询\n")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()

