"""
Neo4j Demo 可视化展示脚本
展示各种查询结果和图表效果
"""

from neo4j import GraphDatabase
import json

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subsection(title):
    """打印子节标题"""
    print(f"\n【{title}】")

def run_query(driver, query, description=None):
    """运行查询并打印结果"""
    if description:
        print_subsection(description)
    
    with driver.session() as session:
        result = session.run(query)
        records = [record for record in result]
        
        if records:
            # 打印表头
            keys = records[0].keys()
            header = " | ".join([f"{k:20}" for k in keys])
            print(header)
            print("-" * len(header))
            
            # 打印数据
            for record in records[:20]:  # 最多显示20条
                row = " | ".join([f"{str(record[k]):20}" for k in keys])
                print(row)
            
            if len(records) > 20:
                print(f"... 还有 {len(records) - 20} 条记录")
        else:
            print("(无结果)")
        
        return records

def main():
    """主函数"""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    try:
        print_section("Neo4j 图数据库 Demo 效果展示")
        
        # 1. 查看所有节点
        print_subsection("1. 数据库概览 - 所有节点")
        run_query(
            driver,
            """
            MATCH (n)
            RETURN labels(n)[0] AS 节点类型, count(n) AS 数量
            ORDER BY 数量 DESC
            """,
            "节点统计"
        )
        
        # 2. 查看所有关系
        print_subsection("2. 数据库概览 - 所有关系")
        run_query(
            driver,
            """
            MATCH ()-[r]->()
            RETURN type(r) AS 关系类型, count(r) AS 数量
            ORDER BY 数量 DESC
            """,
            "关系统计"
        )
        
        # 3. 查看完整的图结构
        print_subsection("3. 图结构可视化 - 所有节点和关系")
        print("提示：在 Neo4j Browser 中运行以下查询可以看到可视化图：")
        print("   MATCH (n) RETURN n LIMIT 25")
        
        # 4. 演员和电影的关系
        print_subsection("4. 演员参演电影关系")
        run_query(
            driver,
            """
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            RETURN p.name AS 演员, m.title AS 电影, r.roles AS 角色
            ORDER BY 演员, 电影
            """,
            "演员参演电影详情"
        )
        
        # 5. 导演和电影的关系
        print_subsection("5. 导演执导电影关系")
        run_query(
            driver,
            """
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            RETURN p.name AS 导演, m.title AS 电影, m.released AS 年份
            ORDER BY 年份
            """,
            "导演执导电影详情"
        )
        
        # 6. 电影详情
        print_subsection("6. 电影详细信息")
        run_query(
            driver,
            """
            MATCH (m:Movie)
            RETURN m.title AS 电影名, m.released AS 年份, m.tagline AS 标语
            ORDER BY 年份
            """,
            "所有电影"
        )
        
        # 7. 演员参演电影数量统计
        print_subsection("7. 演员参演电影数量排名")
        run_query(
            driver,
            """
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            RETURN p.name AS 演员, count(m) AS 参演电影数
            ORDER BY 参演电影数 DESC, 演员
            """,
            "演员参演电影数量统计"
        )
        
        # 8. 电影演员数量统计
        print_subsection("8. 电影演员数量统计")
        run_query(
            driver,
            """
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            RETURN m.title AS 电影, count(p) AS 演员数量
            ORDER BY 演员数量 DESC
            """,
            "每部电影的演员数量"
        )
        
        # 9. 合作关系网络
        print_subsection("9. 演员合作关系网络")
        run_query(
            driver,
            """
            MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
            WHERE p1 <> p2 AND p1.name < p2.name
            RETURN p1.name AS 演员1, p2.name AS 演员2, count(m) AS 合作次数
            ORDER BY 合作次数 DESC, 演员1
            """,
            "演员之间的合作关系"
        )
        
        # 10. 电影制作团队
        print_subsection("10. 电影制作团队（导演+制片人）")
        run_query(
            driver,
            """
            MATCH (m:Movie)
            OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
            OPTIONAL MATCH (pr:Person)-[:PRODUCED]->(m)
            RETURN m.title AS 电影,
                   collect(DISTINCT d.name) AS 导演,
                   collect(DISTINCT pr.name) AS 制片人
            ORDER BY 电影
            """,
            "电影制作团队信息"
        )
        
        # 11. 评论信息
        print_subsection("11. 电影评论信息")
        run_query(
            driver,
            """
            MATCH (p:Person)-[r:REVIEWED]->(m:Movie)
            RETURN p.name AS 评论者, m.title AS 电影, r.rating AS 评分, r.summary AS 评论摘要
            """,
            "电影评论详情"
        )
        
        # 12. 图数据库查询示例
        print_section("图数据库查询示例")
        print("\n以下是一些可以在 Neo4j Browser 中运行的查询示例：\n")
        
        queries = [
            ("查看所有节点和关系", "MATCH (n) RETURN n LIMIT 25"),
            ("查看演员参演电影的关系", "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m"),
            ("查找 Tom Hanks 的所有关系", "MATCH (p:Person {name: 'Tom Hanks'})-[r]->(n) RETURN p, r, n"),
            ("查找《The Matrix》的所有相关人员", "MATCH (m:Movie {title: 'The Matrix'})<-[r]-(p:Person) RETURN m, r, p"),
            ("查找导演执导的所有电影", "MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN p, m"),
        ]
        
        for i, (desc, query) in enumerate(queries, 1):
            print(f"{i}. {desc}:")
            print(f"   {query}\n")
        
        print_section("访问 Neo4j Browser 查看可视化图")
        print("\n🌐 打开浏览器访问: http://localhost:7474")
        print("   用户名: neo4j")
        print("   密码: password\n")
        print("在浏览器中运行上面的查询，可以看到漂亮的图可视化效果！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()

