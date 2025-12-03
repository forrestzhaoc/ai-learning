# Neo4j 图数据库学习项目

这是一个 Neo4j 图数据库的入门学习项目，包含基本的 CRUD 操作和常见查询示例。

## 前置要求

1. **安装 Neo4j 数据库**
   - 方式一：使用 Docker（推荐）
     ```bash
     docker run -d \
       --name neo4j \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password \
       neo4j:latest
     ```
   
   - 方式二：从官网下载安装
     - 访问 https://neo4j.com/download/
     - 下载并安装 Neo4j Desktop 或 Community Edition

2. **安装 Python 依赖**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 配置

在运行 demo 之前，请确保 Neo4j 数据库正在运行，并根据你的配置修改 `demo.py` 中的连接信息：

```python
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
```

## 运行 Demo

### 方式一：使用便捷脚本（推荐）

```bash
# 运行主演示程序
./run_demo.sh demo

# 或直接运行（默认运行主演示）
./run_demo.sh

# 运行详细查询结果展示
./run_demo.sh show

# 运行可视化图结构展示
./run_demo.sh visual
```

### 方式二：直接运行 Python 脚本

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行主演示程序
python3 demo.py

# 运行查询展示
python3 show_demo.py

# 运行可视化展示
python3 visual_demo.py
```

## 项目结构

- `demo.py` - 主要的演示代码，包含 Neo4j 的基本操作
- `show_demo.py` - 详细查询结果展示脚本
- `visual_demo.py` - 可视化图结构展示脚本
- `run_demo.sh` - 便捷启动脚本
- `cypher_examples.cypher` - Cypher 查询示例集合
- `requirements.txt` - Python 依赖包
- `README.md` - 项目说明文档

## 学习内容

本 demo 包含以下内容：

1. **连接数据库** - 如何连接到 Neo4j 数据库
2. **创建节点** - 创建 Person、Movie 等节点
3. **创建关系** - 创建 ACTED_IN、DIRECTED、REVIEWED 等关系
4. **查询数据** - 使用 Cypher 查询语言进行各种查询
5. **更新数据** - 更新节点和关系的属性
6. **删除数据** - 删除节点和关系
7. **复杂查询** - 路径查询、推荐系统等

## Neo4j Browser

启动 Neo4j 后，可以通过浏览器访问：
- URL: http://localhost:7474
- 默认用户名: neo4j
- 默认密码: password（首次登录后会要求修改）

在浏览器中可以：
- 查看图数据可视化
- 执行 Cypher 查询
- 查看数据库统计信息

## 常用 Cypher 查询示例

```cypher
// 查找所有节点
MATCH (n) RETURN n LIMIT 25

// 查找所有 Person 节点
MATCH (p:Person) RETURN p

// 查找特定演员参演的电影
MATCH (p:Person {name: "Tom Hanks"})-[:ACTED_IN]->(m:Movie)
RETURN m.title

// 查找两个演员之间的最短路径
MATCH path = shortestPath(
  (p1:Person {name: "Tom Hanks"})-[*]-(p2:Person {name: "Keanu Reeves"})
)
RETURN path
```

## 参考资料

- [Neo4j 官方文档](https://neo4j.com/docs/)
- [Cypher 查询语言](https://neo4j.com/developer/cypher/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)

