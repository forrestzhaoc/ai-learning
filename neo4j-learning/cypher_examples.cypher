// Neo4j Cypher 查询示例
// 这些查询可以在 Neo4j Browser (http://localhost:7474) 中运行

// ============================================
// 基础查询
// ============================================

// 1. 查找所有节点（限制25个）
MATCH (n) RETURN n LIMIT 25

// 2. 查找所有 Person 节点
MATCH (p:Person) RETURN p

// 3. 查找所有 Movie 节点
MATCH (m:Movie) RETURN m

// 4. 查找特定名称的 Person
MATCH (p:Person {name: "Tom Hanks"}) RETURN p

// ============================================
// 关系查询
// ============================================

// 5. 查找 Tom Hanks 参演的所有电影
MATCH (p:Person {name: "Tom Hanks"})-[:ACTED_IN]->(m:Movie)
RETURN m.title AS title, m.released AS released
ORDER BY m.released

// 6. 查找《The Matrix》的所有演员
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {title: "The Matrix"})
RETURN p.name AS actor, r.roles AS roles

// 7. 查找《The Matrix》的导演
MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: "The Matrix"})
RETURN p.name AS director

// 8. 查找所有关系
MATCH ()-[r]->() RETURN r LIMIT 25

// ============================================
// 复杂查询
// ============================================

// 9. 查找与 Keanu Reeves 合作过的演员
MATCH (p1:Person {name: "Keanu Reeves"})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE p1 <> p2
RETURN DISTINCT p2.name AS co_actor
ORDER BY co_actor

// 10. 查找两个演员之间的最短路径
MATCH path = shortestPath(
  (p1:Person {name: "Tom Hanks"})-[*]-(p2:Person {name: "Keanu Reeves"})
)
RETURN path

// 11. 查找所有路径（最多3跳）
MATCH path = (p1:Person {name: "Tom Hanks"})-[*1..3]-(p2:Person {name: "Keanu Reeves"})
RETURN path
LIMIT 10

// 12. 查找参演电影数量最多的演员
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, count(m) AS movie_count
ORDER BY movie_count DESC
LIMIT 10

// 13. 查找被最多演员参演的电影
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN m.title AS movie, count(p) AS actor_count
ORDER BY actor_count DESC
LIMIT 10

// ============================================
// 聚合和统计
// ============================================

// 14. 统计每种类型的节点数量
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS count
ORDER BY count DESC

// 15. 统计每种类型的关系数量
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY count DESC

// 16. 查找所有节点的平均属性值（如果有数值属性）
MATCH (p:Person)
WHERE p.born IS NOT NULL
RETURN avg(p.born) AS average_birth_year

// ============================================
// 模式匹配
// ============================================

// 17. 查找导演和演员的关系模式
MATCH (director:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(actor:Person)
RETURN director.name AS director, m.title AS movie, actor.name AS actor
LIMIT 20

// 18. 查找三部曲电影（同一导演的多部电影）
MATCH (director:Person)-[:DIRECTED]->(m:Movie)
WITH director, collect(m.title) AS movies
WHERE size(movies) >= 2
RETURN director.name AS director, movies

// ============================================
// 推荐系统示例
// ============================================

// 19. 基于共同参演电影的推荐：推荐与 Tom Hanks 有相似电影品味的演员
MATCH (p1:Person {name: "Tom Hanks"})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE p1 <> p2
WITH p2, count(m) AS common_movies
ORDER BY common_movies DESC
RETURN p2.name AS recommended_actor, common_movies AS common_movie_count
LIMIT 10

// 20. 查找可能喜欢的电影：查找与 Tom Hanks 合作过的演员参演的其他电影
MATCH (p1:Person {name: "Tom Hanks"})-[:ACTED_IN]->(m1:Movie)<-[:ACTED_IN]-(p2:Person)-[:ACTED_IN]->(m2:Movie)
WHERE p1 <> p2 AND NOT (p1)-[:ACTED_IN]->(m2)
RETURN DISTINCT m2.title AS recommended_movie, count(p2) AS recommendation_score
ORDER BY recommendation_score DESC
LIMIT 10

// ============================================
// 更新操作
// ============================================

// 21. 更新节点属性
MATCH (p:Person {name: "Tom Hanks"})
SET p.awards = "Oscar Winner"
RETURN p

// 22. 添加新属性
MATCH (m:Movie {title: "The Matrix"})
SET m.genre = "Science Fiction"
RETURN m

// 23. 更新关系属性
MATCH (p:Person {name: "Keanu Reeves"})-[r:ACTED_IN]->(m:Movie {title: "The Matrix"})
SET r.salary = 10000000
RETURN r

// ============================================
// 删除操作（谨慎使用！）
// ============================================

// 24. 删除节点及其所有关系
// MATCH (p:Person {name: "Some Person"}) DETACH DELETE p

// 25. 删除特定关系
// MATCH (p:Person {name: "Some Person"})-[r:ACTED_IN]->(m:Movie {title: "Some Movie"})
// DELETE r

// 26. 删除所有节点和关系（清空数据库）
// MATCH (n) DETACH DELETE n

// ============================================
// 创建操作
// ============================================

// 27. 创建新节点
// CREATE (p:Person {name: "New Actor", born: 1990})

// 28. 创建关系
// MATCH (p:Person {name: "New Actor"})
// MATCH (m:Movie {title: "The Matrix"})
// CREATE (p)-[:ACTED_IN {roles: ["Extra"]}]->(m)

// 29. 使用 MERGE 避免重复创建（如果不存在则创建，存在则匹配）
// MERGE (p:Person {name: "New Actor"})
// ON CREATE SET p.created = timestamp()
// ON MATCH SET p.last_seen = timestamp()
// RETURN p

