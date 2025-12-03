"""
Neo4j 图数据库入门 Demo

这个 demo 演示了 Neo4j 的基本操作，包括：
- 连接数据库
- 创建节点和关系
- 查询数据
- 更新和删除操作
"""

from neo4j import GraphDatabase
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Neo4jDemo:
    """Neo4j 演示类"""
    
    def __init__(self, uri: str, user: str = None, password: str = None):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j 数据库 URI，例如 "bolt://localhost:7687"
            user: 用户名（可选）
            password: 密码（可选）
        """
        # Neo4j驱动需要auth参数，即使认证已禁用
        if user and password:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        else:
            # 使用默认用户名和空密码（当认证禁用时）
            self.driver = GraphDatabase.driver(uri, auth=("neo4j", "neo4j"))
        logger.info(f"已连接到 Neo4j 数据库: {uri}")
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        logger.info("数据库连接已关闭")
    
    def clear_database(self):
        """清空数据库（用于演示）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("数据库已清空")
    
    def create_person(self, name: str, born: int):
        """创建 Person 节点"""
        with self.driver.session() as session:
            result = session.run(
                "CREATE (p:Person {name: $name, born: $born}) RETURN p",
                name=name, born=born
            )
            logger.info(f"创建 Person 节点: {name} (born: {born})")
            return result.single()
    
    def create_movie(self, title: str, released: int, tagline: str = None):
        """创建 Movie 节点"""
        with self.driver.session() as session:
            query = "CREATE (m:Movie {title: $title, released: $released"
            params = {"title": title, "released": released}
            
            if tagline:
                query += ", tagline: $tagline"
                params["tagline"] = tagline
            
            query += "}) RETURN m"
            
            result = session.run(query, **params)
            logger.info(f"创建 Movie 节点: {title} (released: {released})")
            return result.single()
    
    def create_acted_in_relationship(self, person_name: str, movie_title: str, roles: list):
        """创建 ACTED_IN 关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $person_name})
                MATCH (m:Movie {title: $movie_title})
                CREATE (p)-[r:ACTED_IN {roles: $roles}]->(m)
                RETURN r
                """,
                person_name=person_name,
                movie_title=movie_title,
                roles=roles
            )
            logger.info(f"创建关系: {person_name} ACTED_IN {movie_title} (roles: {roles})")
            return result.single()
    
    def create_directed_relationship(self, person_name: str, movie_title: str):
        """创建 DIRECTED 关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $person_name})
                MATCH (m:Movie {title: $movie_title})
                CREATE (p)-[r:DIRECTED]->(m)
                RETURN r
                """,
                person_name=person_name,
                movie_title=movie_title
            )
            logger.info(f"创建关系: {person_name} DIRECTED {movie_title}")
            return result.single()
    
    def create_reviewed_relationship(self, person_name: str, movie_title: str, rating: int, summary: str):
        """创建 REVIEWED 关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $person_name})
                MATCH (m:Movie {title: $movie_title})
                CREATE (p)-[r:REVIEWED {rating: $rating, summary: $summary}]->(m)
                RETURN r
                """,
                person_name=person_name,
                movie_title=movie_title,
                rating=rating,
                summary=summary
            )
            logger.info(f"创建关系: {person_name} REVIEWED {movie_title} (rating: {rating})")
            return result.single()
    
    def find_person(self, name: str):
        """查找 Person 节点"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Person {name: $name}) RETURN p",
                name=name
            )
            record = result.single()
            if record:
                person = record["p"]
                logger.info(f"找到 Person: {person['name']} (born: {person.get('born', 'N/A')})")
                return person
            else:
                logger.info(f"未找到 Person: {name}")
                return None
    
    def find_movies_by_actor(self, actor_name: str):
        """查找演员参演的所有电影"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $name})-[:ACTED_IN]->(m:Movie)
                RETURN m.title AS title, m.released AS released, m.tagline AS tagline
                ORDER BY m.released
                """,
                name=actor_name
            )
            movies = [record for record in result]
            logger.info(f"{actor_name} 参演了 {len(movies)} 部电影:")
            for movie in movies:
                logger.info(f"  - {movie['title']} ({movie['released']})")
            return movies
    
    def find_actors_in_movie(self, movie_title: str):
        """查找电影中的所有演员"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {title: $title})
                RETURN p.name AS name, r.roles AS roles
                """,
                title=movie_title
            )
            actors = [record for record in result]
            logger.info(f"《{movie_title}》的演员:")
            for actor in actors:
                roles_str = ", ".join(actor['roles']) if actor['roles'] else "N/A"
                logger.info(f"  - {actor['name']} (roles: {roles_str})")
            return actors
    
    def find_movie_director(self, movie_title: str):
        """查找电影的导演"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: $title})
                RETURN p.name AS name
                """,
                title=movie_title
            )
            directors = [record for record in result]
            logger.info(f"《{movie_title}》的导演:")
            for director in directors:
                logger.info(f"  - {director['name']}")
            return directors
    
    def find_co_actors(self, actor_name: str):
        """查找与指定演员合作过的其他演员"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p1:Person {name: $name})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
                WHERE p1 <> p2
                RETURN DISTINCT p2.name AS name
                ORDER BY name
                """,
                name=actor_name
            )
            co_actors = [record for record in result]
            logger.info(f"与 {actor_name} 合作过的演员 ({len(co_actors)} 人):")
            for actor in co_actors[:10]:  # 只显示前10个
                logger.info(f"  - {actor['name']}")
            return co_actors
    
    def find_shortest_path(self, person1: str, person2: str):
        """查找两个 Person 之间的最短路径"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (p1:Person {name: $name1})-[*]-(p2:Person {name: $name2})
                )
                RETURN path
                """,
                name1=person1,
                name2=person2
            )
            record = result.single()
            if record:
                path = record["path"]
                logger.info(f"{person1} 和 {person2} 之间的最短路径:")
                logger.info(f"  路径长度: {len(path.relationships)}")
                return path
            else:
                logger.info(f"未找到 {person1} 和 {person2} 之间的路径")
                return None
    
    def get_statistics(self):
        """获取数据库统计信息"""
        with self.driver.session() as session:
            # 节点统计
            node_result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
                """
            )
            logger.info("=== 节点统计 ===")
            for record in node_result:
                logger.info(f"  {record['label']}: {record['count']} 个")
            
            # 关系统计
            rel_result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
                """
            )
            logger.info("=== 关系统计 ===")
            for record in rel_result:
                logger.info(f"  {record['type']}: {record['count']} 个")
    
    def update_person_property(self, name: str, property_name: str, property_value):
        """更新 Person 节点的属性"""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (p:Person {{name: $name}}) SET p.{property_name} = $value RETURN p",
                name=name,
                value=property_value
            )
            logger.info(f"更新 {name} 的 {property_name} 为 {property_value}")
            return result.single()
    
    def delete_person(self, name: str):
        """删除 Person 节点及其所有关系"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Person {name: $name}) DETACH DELETE p RETURN count(p) AS deleted",
                name=name
            )
            deleted = result.single()["deleted"]
            logger.info(f"删除了 {deleted} 个 Person 节点: {name}")
            return deleted


def main():
    """主函数：运行大规模数据演示"""
    import random
    
    # 数据库连接配置
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "password"
    
    # 创建 Neo4j 实例
    neo4j_demo = Neo4jDemo(URI, USER, PASSWORD)
    
    try:
        logger.info("=" * 80)
        logger.info("Neo4j 大规模图数据库 Demo - 展示图数据库优势")
        logger.info("=" * 80)
        
        # 清空数据库
        logger.info("\n1. 清空数据库...")
        neo4j_demo.clear_database()
        
        # 生成大规模数据
        logger.info("\n2. 开始生成大规模数据...")
        
        # 生成演员名字列表
        first_names = ["Tom", "John", "Emma", "Chris", "Jennifer", "Michael", "Sarah", "David", "Lisa", "Robert",
                       "Mary", "James", "Patricia", "William", "Linda", "Richard", "Barbara", "Joseph", "Elizabeth", "Thomas",
                       "Jessica", "Charles", "Susan", "Christopher", "Karen", "Daniel", "Nancy", "Matthew", "Betty", "Anthony",
                       "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew",
                       "Donna", "Joshua", "Michelle", "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Melissa", "George"]
        
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                      "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
                      "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
                      "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
                      "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts", "Gomez", "Phillips"]
        
        genres = ["Action", "Comedy", "Drama", "Thriller", "Horror", "Sci-Fi", "Romance", "Adventure", "Fantasy", "Crime",
                  "Mystery", "Animation", "Documentary", "War", "Western", "Musical", "Biography", "History", "Sport", "Family"]
        
        taglines = [
            "A story of courage and determination", "Where dreams become reality", "The ultimate adventure begins",
            "Love conquers all", "In a world where nothing is as it seems", "The fight for freedom starts now",
            "One man's journey to redemption", "When the past meets the future", "The truth will set you free",
            "A tale of friendship and betrayal", "Beyond the limits of imagination", "The battle for survival",
            "Where heroes are made", "A journey into the unknown", "The power of hope", "When everything changes",
            "The ultimate test of will", "A story that will change everything", "The beginning of the end",
            "Where legends are born", "The price of freedom", "A fight for justice", "The power of love",
            "When worlds collide", "The search for truth", "A story of triumph", "The road to redemption"
        ]
        
        awards_list = ["Oscar Winner", "Golden Globe", "Emmy Award", "BAFTA", "Cannes", "Sundance", "Academy Award"]
        nationalities = ["American", "British", "Canadian", "Australian", "French", "German", "Italian", "Spanish", "Japanese", "Chinese"]
        
        # 生成500+演员
        logger.info("\n3. 创建 500+ 个演员节点...")
        actors = []
        with neo4j_demo.driver.session() as session:
            for i in range(550):
                first = random.choice(first_names)
                last = random.choice(last_names)
                name = f"{first} {last}"
                born = random.randint(1950, 2000)
                nationality = random.choice(nationalities)
                awards = random.choice(awards_list) if random.random() < 0.3 else None
                
                query = "CREATE (p:Person {name: $name, born: $born, nationality: $nationality"
                params = {"name": name, "born": born, "nationality": nationality}
                
                if awards:
                    query += ", awards: $awards"
                    params["awards"] = awards
                
                query += "}) RETURN p"
                
                session.run(query, **params)
                actors.append(name)
                if (i + 1) % 50 == 0:
                    logger.info(f"  已创建 {i + 1} 个演员...")
        
        logger.info(f"✓ 共创建 {len(actors)} 个演员")
        
        # 生成2000+电影
        logger.info("\n4. 创建 2000+ 部电影节点...")
        movies = []
        movie_titles = []
        with neo4j_demo.driver.session() as session:
            for i in range(2100):
                # 生成电影标题
                title_words = ["The", "A", "In", "Beyond", "Return", "Rise", "Fall", "Last", "First", "Final",
                              "Secret", "Dark", "Light", "Lost", "Found", "Hidden", "Eternal", "Legend", "Quest", "Journey"]
                noun_words = ["Hero", "Warrior", "Kingdom", "Empire", "City", "Storm", "Fire", "Ice", "Shadow", "Light",
                             "Dream", "Night", "Day", "Dawn", "Dusk", "Star", "Moon", "Sun", "Ocean", "Mountain",
                             "River", "Forest", "Desert", "Battle", "War", "Peace", "Love", "Hope", "Fear", "Courage"]
                
                title = f"{random.choice(title_words)} {random.choice(noun_words)}"
                if random.random() < 0.3:
                    title += f" {random.randint(1, 5)}"
                
                released = random.randint(1980, 2024)
                genre = random.choice(genres)
                tagline = random.choice(taglines)
                budget = random.randint(1000000, 200000000)
                box_office = int(budget * random.uniform(0.5, 5.0))
                rating = round(random.uniform(5.0, 9.5), 1)
                
                query = """
                CREATE (m:Movie {
                    title: $title,
                    released: $released,
                    genre: $genre,
                    tagline: $tagline,
                    budget: $budget,
                    box_office: $box_office,
                    rating: $rating
                })
                RETURN m
                """
                
                session.run(query, title=title, released=released, genre=genre, tagline=tagline,
                           budget=budget, box_office=box_office, rating=rating)
                movies.append(title)
                movie_titles.append(title)
                
                if (i + 1) % 200 == 0:
                    logger.info(f"  已创建 {i + 1} 部电影...")
        
        logger.info(f"✓ 共创建 {len(movies)} 部电影")
        
        # 创建大量 ACTED_IN 关系（每个演员参演多部电影）
        logger.info("\n5. 创建 ACTED_IN 关系（每个演员参演 3-15 部电影）...")
        with neo4j_demo.driver.session() as session:
            total_acted = 0
            for actor in actors:
                num_movies = random.randint(3, 15)
                selected_movies = random.sample(movie_titles, min(num_movies, len(movie_titles)))
                
                for movie_title in selected_movies:
                    roles = [f"Character {random.randint(1, 10)}"]
                    if random.random() < 0.3:
                        roles.append(f"Supporting Role {random.randint(1, 5)}")
                    
                    session.run("""
                        MATCH (p:Person {name: $actor})
                        MATCH (m:Movie {title: $movie})
                        CREATE (p)-[r:ACTED_IN {roles: $roles, salary: $salary}]->(m)
                        """,
                        actor=actor, movie=movie_title, roles=roles,
                        salary=random.randint(100000, 10000000)
                    )
                    total_acted += 1
                
                if actors.index(actor) % 50 == 0:
                    logger.info(f"  已处理 {actors.index(actor) + 1}/{len(actors)} 个演员的关系...")
        
        logger.info(f"✓ 共创建 {total_acted} 个 ACTED_IN 关系")
        
        # 创建 DIRECTED 关系（部分演员也是导演）
        logger.info("\n6. 创建 DIRECTED 关系...")
        directors = random.sample(actors, min(150, len(actors)))
        with neo4j_demo.driver.session() as session:
            total_directed = 0
            for director in directors:
                num_movies = random.randint(1, 8)
                selected_movies = random.sample(movie_titles, min(num_movies, len(movie_titles)))
                
                for movie_title in selected_movies:
                    session.run("""
                        MATCH (p:Person {name: $director})
                        MATCH (m:Movie {title: $movie})
                        CREATE (p)-[r:DIRECTED {year: $year}]->(m)
                        """,
                        director=director, movie=movie_title,
                        year=random.randint(1980, 2024)
                    )
                    total_directed += 1
            
            logger.info(f"✓ 共创建 {total_directed} 个 DIRECTED 关系")
        
        # 创建 PRODUCED 关系
        logger.info("\n7. 创建 PRODUCED 关系...")
        producers = random.sample(actors, min(100, len(actors)))
        with neo4j_demo.driver.session() as session:
            total_produced = 0
            for producer in producers:
                num_movies = random.randint(1, 10)
                selected_movies = random.sample(movie_titles, min(num_movies, len(movie_titles)))
                
                for movie_title in selected_movies:
                    session.run("""
                        MATCH (p:Person {name: $producer})
                        MATCH (m:Movie {title: $movie})
                        CREATE (p)-[r:PRODUCED {budget: $budget}]->(m)
                        """,
                        producer=producer, movie=movie_title,
                        budget=random.randint(5000000, 150000000)
                    )
                    total_produced += 1
            
            logger.info(f"✓ 共创建 {total_produced} 个 PRODUCED 关系")
        
        # 创建 REVIEWED 关系
        logger.info("\n8. 创建 REVIEWED 关系...")
        reviewers = random.sample(actors, min(200, len(actors)))
        with neo4j_demo.driver.session() as session:
            total_reviewed = 0
            for reviewer in reviewers:
                num_reviews = random.randint(5, 30)
                selected_movies = random.sample(movie_titles, min(num_reviews, len(movie_titles)))
                
                for movie_title in selected_movies:
                    session.run("""
                        MATCH (p:Person {name: $reviewer})
                        MATCH (m:Movie {title: $movie})
                        CREATE (p)-[r:REVIEWED {
                            rating: $rating,
                            summary: $summary,
                            date: $date
                        }]->(m)
                        """,
                        reviewer=reviewer, movie=movie_title,
                        rating=random.randint(1, 10),
                        summary=f"Review summary for {movie_title}",
                        date=f"{random.randint(2000, 2024)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                    )
                    total_reviewed += 1
            
            logger.info(f"✓ 共创建 {total_reviewed} 个 REVIEWED 关系")
        
        # 创建 FRIENDS_WITH 关系（演员之间的友谊）
        logger.info("\n9. 创建 FRIENDS_WITH 关系（演员社交网络）...")
        with neo4j_demo.driver.session() as session:
            total_friends = 0
            for actor in actors[:300]:  # 前300个演员
                num_friends = random.randint(5, 20)
                friends = random.sample([a for a in actors if a != actor], min(num_friends, len(actors) - 1))
                
                for friend in friends:
                    session.run("""
                        MATCH (p1:Person {name: $actor})
                        MATCH (p2:Person {name: $friend})
                        MERGE (p1)-[r:FRIENDS_WITH {since: $year}]-(p2)
                        """,
                        actor=actor, friend=friend,
                        year=random.randint(1990, 2020)
                    )
                    total_friends += 1
            
            logger.info(f"✓ 共创建 {total_friends} 个 FRIENDS_WITH 关系")
        
        # 创建 WORKED_WITH 关系（通过电影合作）
        logger.info("\n10. 创建 WORKED_WITH 关系（基于共同参演）...")
        with neo4j_demo.driver.session() as session:
            session.run("""
                MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
                WHERE p1 <> p2 AND p1.name < p2.name
                WITH p1, p2, count(m) AS co_movies
                MERGE (p1)-[r:WORKED_WITH {movies_together: co_movies}]-(p2)
                """)
            logger.info("✓ WORKED_WITH 关系创建完成")
        
        # 创建 MARRIED_TO 关系
        logger.info("\n11. 创建 MARRIED_TO 关系...")
        with neo4j_demo.driver.session() as session:
            couples = random.sample(list(zip(actors[:100], actors[100:200])), 30)
            for actor1, actor2 in couples:
                session.run("""
                    MATCH (p1:Person {name: $actor1})
                    MATCH (p2:Person {name: $actor2})
                    CREATE (p1)-[r:MARRIED_TO {since: $year}]->(p2)
                    """,
                    actor1=actor1, actor2=actor2,
                    year=random.randint(1990, 2020)
                )
            logger.info("✓ 创建了 30 个 MARRIED_TO 关系")
        
        # 创建 AWARDED 关系（获奖关系）
        logger.info("\n12. 创建 AWARDED 关系...")
        award_types = ["Best Actor", "Best Actress", "Best Director", "Best Picture", "Best Supporting Actor"]
        with neo4j_demo.driver.session() as session:
            for i in range(500):
                actor = random.choice(actors)
                movie = random.choice(movie_titles)
                award = random.choice(award_types)
                year = random.randint(1990, 2024)
                
                session.run("""
                    MATCH (p:Person {name: $actor})
                    MATCH (m:Movie {title: $movie})
                    CREATE (p)-[r:AWARDED {award: $award, year: $year}]->(m)
                    """,
                    actor=actor, movie=movie, award=award, year=year
                )
            logger.info("✓ 创建了 500 个 AWARDED 关系")
        
        # 统计信息
        logger.info("\n13. 数据库统计信息:")
        neo4j_demo.get_statistics()
        
        logger.info("\n" + "=" * 80)
        logger.info("大规模数据生成完成！")
        logger.info("=" * 80)
        logger.info("\n数据规模:")
        logger.info(f"  - 演员: 550+ 人")
        logger.info(f"  - 电影: 2100+ 部")
        logger.info(f"  - 关系类型: ACTED_IN, DIRECTED, PRODUCED, REVIEWED, FRIENDS_WITH, WORKED_WITH, MARRIED_TO, AWARDED")
        logger.info("\n提示：")
        logger.info("1. 打开 Neo4j Browser (http://localhost:7474) 查看可视化图")
        logger.info("2. 尝试运行复杂查询来体验图数据库的优势：")
        logger.info("   - 查找演员合作网络")
        logger.info("   - 查找最短路径")
        logger.info("   - 社区发现")
        logger.info("   - 影响力分析")
        
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
    finally:
        neo4j_demo.close()


if __name__ == "__main__":
    main()

