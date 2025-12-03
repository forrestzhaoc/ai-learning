# Neo4j Demo 快速开始指南

## ✅ Demo 已成功运行！

### 📊 Demo 效果总结

**数据统计：**
- **节点**: 9 个 Person（人物）+ 5 个 Movie（电影）= 14 个节点
- **关系**: 19 个关系（11个参演 + 6个执导 + 1个制作 + 1个评论）

**主要功能演示：**
1. ✅ 创建节点和关系
2. ✅ 查询演员参演的电影
3. ✅ 查询电影的演员和导演
4. ✅ 查找演员之间的合作关系
5. ✅ 更新节点属性
6. ✅ 统计查询

### 🚀 查看 Demo 效果

#### 方式 1: 命令行查看（已完成）
```bash
cd /home/ubuntu/projects/ai-learning/neo4j-learning
source venv/bin/activate
python show_demo.py
```

#### 方式 2: 浏览器可视化（推荐）
1. **打开 Neo4j Browser**
   - 访问: http://localhost:7474
   - 用户名: `neo4j`
   - 密码: `password`

2. **运行可视化查询**
   在浏览器顶部的查询框中输入以下查询：

   ```cypher
   // 查看所有节点和关系（可视化图）
   MATCH (n) RETURN n LIMIT 25
   ```

   ```cypher
   // 查看演员和电影的关系
   MATCH (p:Person)-[:ACTED_IN]->(m:Movie) 
   RETURN p, m
   ```

   ```cypher
   // 查看《The Matrix》的所有相关人员
   MATCH (m:Movie {title: 'The Matrix'})<-[r]-(p:Person) 
   RETURN m, r, p
   ```

   ```cypher
   // 查看 Tom Hanks 的所有关系
   MATCH (p:Person {name: 'Tom Hanks'})-[r]->(n) 
   RETURN p, r, n
   ```

### 📁 项目文件说明

- `demo.py` - 主演示程序，包含完整的 CRUD 操作
- `show_demo.py` - 可视化展示脚本，展示各种查询结果
- `cypher_examples.cypher` - 29 个 Cypher 查询示例
- `requirements.txt` - Python 依赖包
- `README.md` - 完整项目文档

### 🎯 下一步学习

1. **运行更多查询**
   - 查看 `cypher_examples.cypher` 文件
   - 在 Neo4j Browser 中尝试不同的查询

2. **修改和扩展**
   - 修改 `demo.py` 添加更多数据
   - 创建自己的图数据库应用

3. **学习 Cypher 查询语言**
   - 参考 Neo4j 官方文档: https://neo4j.com/docs/cypher-manual/
   - 练习文件中的查询示例

### 💡 常用命令

```bash
# 运行主 demo
python demo.py

# 运行可视化展示
python show_demo.py

# 检查 Neo4j 状态
sudo systemctl status neo4j

# 重启 Neo4j
sudo systemctl restart neo4j
```

### 🌐 访问信息

- **Neo4j Browser**: http://localhost:7474
- **Bolt 连接**: bolt://localhost:7687
- **默认用户名**: neo4j
- **默认密码**: password

---

**享受学习 Neo4j 图数据库！** 🎉

