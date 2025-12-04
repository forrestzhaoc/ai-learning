# 快速开始指南

## Milvus 向量数据库实现图数据库场景

### 1. 安装依赖

```bash
cd /home/ubuntu/projects/ai-learning/vector-db-learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 运行演示

#### 基础演示

```bash
python3 demo.py
```

这将展示：
- 节点和关系的向量化存储
- 相似度搜索
- 关系查询
- 路径查找
- 推荐系统

#### 数据挖掘查询

```bash
python3 mining_queries.py
```

这将展示：
- 演员合作网络分析
- 电影推荐系统
- 路径分析
- 影响力分析
- 社区发现
- 时间序列分析
- 属性挖掘
- 关系强度分析

### 3. 使用 Milvus Lite（推荐）

默认使用 Milvus Lite 模式，无需启动任何服务：

```python
from implementations.milvus_impl import MilvusGraphDB

# 自动使用 Milvus Lite
db = MilvusGraphDB(use_lite=True)
```

### 4. 使用 Milvus 服务模式

如果需要使用 Milvus 服务（适合生产环境）：

#### 启动 Milvus 服务（Docker）

```bash
docker run -d \
  --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

#### 修改代码

```python
from implementations.milvus_impl import MilvusGraphDB

# 连接到 Milvus 服务
db = MilvusGraphDB(
    host="localhost",
    port=19530,
    use_lite=False
)
```

### 5. 代码示例

```python
from implementations.milvus_impl import MilvusGraphDB
from models.graph_models import Node, Relationship

# 初始化数据库
db = MilvusGraphDB(use_lite=True)

# 创建节点
node = Node("p1", "Person", {"name": "Tom Hanks", "born": 1956})
db.add_node(node)

# 查找相似节点
similar_nodes = db.find_similar_nodes("Tom Hanks", top_k=5)
for node in similar_nodes:
    print(f"{node['properties']['name']} - 相似度: {node['similarity']:.4f}")

# 查找关系
relationships = db.find_relationships(source_id="p1", rel_type="ACTED_IN")
for rel in relationships:
    print(f"关系: {rel['type']}")
```

### 6. 常见问题

#### Q: Milvus Lite 安装失败？

A: 确保使用最新版本的 pymilvus：
```bash
pip install --upgrade pymilvus
```

#### Q: 如何清空数据库？

A: 使用 `clear()` 方法：
```python
db.clear()
```

#### Q: 如何切换不同的向量数据库？

A: 修改 `demo.py` 中的导入：
```python
# 使用 Milvus
from implementations.milvus_impl import MilvusGraphDB

# 或使用 ChromaDB
# from implementations.chroma_impl import ChromaGraphDB
```

### 7. 性能优化

- **批量插入**：对于大量数据，考虑批量插入以提高性能
- **索引优化**：根据数据规模调整索引参数
- **缓存**：节点信息已自动缓存，减少查询次数

### 8. 下一步

- 查看 `demo.py` 了解基础用法
- 查看 `mining_queries.py` 了解复杂查询
- 查看 `README.md` 了解详细文档



