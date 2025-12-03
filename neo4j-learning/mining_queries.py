"""
Neo4j 数据挖掘查询示例
展示各种信息挖掘的 Cypher 查询
"""

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def run_mining_query(driver, query, description):
    """运行挖掘查询并显示结果"""
    print(f"\n【{description}】")
    print("-" * 80)
    
    with driver.session() as session:
        result = session.run(query)
        records = [record for record in result]
        
        if records:
            keys = records[0].keys()
            # 打印表头
            header = " | ".join([f"{k:25}" for k in keys])
            print(header)
            print("-" * len(header))
            
            # 打印数据
            for record in records[:15]:  # 最多显示15条
                row = " | ".join([f"{str(record[k])[:25]:25}" for k in keys])
                print(row)
            
            if len(records) > 15:
                print(f"... 还有 {len(records) - 15} 条记录")
        else:
            print("(无结果)")
        
        return records

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    try:
        print_section("Neo4j 数据挖掘分析")
        
        # 1. 演员合作网络分析
        print_section("1. 演员合作网络分析")
        
        run_mining_query(
            driver,
            """
            MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
            WHERE p1 <> p2 AND p1.name < p2.name
            RETURN p1.name AS 演员1, p2.name AS 演员2, 
                   count(m) AS 合作次数,
                   collect(m.title) AS 合作电影
            ORDER BY 合作次数 DESC
            """,
            "演员合作频率排名"
        )
        
        # 2. 电影推荐系统
        print_section("2. 电影推荐系统")
        
        run_mining_query(
            driver,
            """
            // 基于共同参演演员推荐电影
            MATCH (p1:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m1:Movie)<-[:ACTED_IN]-(p2:Person)-[:ACTED_IN]->(m2:Movie)
            WHERE p1 <> p2 AND NOT (p1)-[:ACTED_IN]->(m2)
            RETURN DISTINCT m2.title AS 推荐电影, 
                   count(DISTINCT p2) AS 共同演员数
            ORDER BY 共同演员数 DESC
            LIMIT 10
            """,
            "基于 Tom Hanks 的电影推荐（通过共同演员）"
        )
        
        # 3. 路径分析
        print_section("3. 路径分析")
        
        run_mining_query(
            driver,
            """
            // 找出两个演员之间的所有路径（最多2跳）
            MATCH path = (p1:Person {name: 'Tom Hanks'})-[*1..2]-(p2:Person {name: 'Keanu Reeves'})
            WHERE p1 <> p2
            RETURN length(path) AS 路径长度,
                   [n in nodes(path) | n.name] AS 路径节点
            LIMIT 5
            """,
            "Tom Hanks 和 Keanu Reeves 之间的路径"
        )
        
        # 4. 影响力分析
        print_section("4. 影响力分析")
        
        run_mining_query(
            driver,
            """
            // 度中心性：连接最多的节点
            MATCH (p:Person)-[r]->()
            RETURN p.name AS 人物, 
                   count(r) AS 连接数,
                   labels(p)[0] AS 类型
            ORDER BY 连接数 DESC
            """,
            "连接数排名（影响力分析）"
        )
        
        run_mining_query(
            driver,
            """
            // 中介中心性：作为桥梁连接不同群体
            MATCH (bridge:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(other:Person)
            WHERE bridge <> other
            WITH bridge, count(DISTINCT other) AS 连接的不同演员数
            RETURN bridge.name AS 桥梁人物, 连接的不同演员数
            ORDER BY 连接的不同演员数 DESC
            """,
            "桥梁人物分析（连接不同演员）"
        )
        
        # 5. 社区发现
        print_section("5. 社区发现")
        
        run_mining_query(
            driver,
            """
            // 识别电影系列（相同导演的电影）
            MATCH (d:Person)-[:DIRECTED]->(m:Movie)
            WITH d, collect(m.title) AS 电影列表
            WHERE size(电影列表) > 1
            RETURN d.name AS 导演, 电影列表, size(电影列表) AS 电影数量
            ORDER BY 电影数量 DESC
            """,
            "导演的电影系列（社区）"
        )
        
        run_mining_query(
            driver,
            """
            // 识别演员群体（经常合作的演员）
            MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
            WHERE p1 <> p2 AND p1.name < p2.name
            WITH p1, p2, count(m) AS 合作次数
            WHERE 合作次数 >= 2
            RETURN p1.name AS 演员1, p2.name AS 演员2, 合作次数
            ORDER BY 合作次数 DESC
            """,
            "紧密合作的演员群体"
        )
        
        # 6. 时间序列分析
        print_section("6. 时间序列分析")
        
        run_mining_query(
            driver,
            """
            // 演员职业生涯轨迹
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            RETURN p.name AS 演员, 
                   min(m.released) AS 首部电影年份,
                   max(m.released) AS 最新电影年份,
                   count(m) AS 参演电影数
            ORDER BY 首部电影年份
            """,
            "演员职业生涯时间线"
        )
        
        # 7. 属性挖掘
        print_section("7. 属性挖掘")
        
        run_mining_query(
            driver,
            """
            // 电影年份分布
            MATCH (m:Movie)
            RETURN m.released AS 年份, count(*) AS 电影数量
            ORDER BY 年份
            """,
            "电影年份分布"
        )
        
        run_mining_query(
            driver,
            """
            // 演员年龄与参演电影的关系
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            WHERE p.born IS NOT NULL
            RETURN p.name AS 演员,
                   p.born AS 出生年份,
                   avg(m.released - p.born) AS 平均参演年龄,
                   count(m) AS 参演电影数
            ORDER BY 平均参演年龄
            """,
            "演员参演年龄分析"
        )
        
        # 8. 关系强度分析
        print_section("8. 关系强度分析")
        
        run_mining_query(
            driver,
            """
            // 导演-演员合作强度
            MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person)
            RETURN d.name AS 导演, 
                   a.name AS 演员,
                   count(m) AS 合作次数,
                   collect(m.title) AS 合作电影
            ORDER BY 合作次数 DESC
            """,
            "导演-演员合作强度"
        )
        
        print_section("挖掘分析完成")
        print("\n💡 提示：这些查询可以在 Neo4j Browser 中运行，查看可视化效果")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()

