# 向量数据库实现图数据库场景

这个项目展示了如何使用向量数据库来实现类似 Neo4j 图数据库的场景。我们将使用向量嵌入来表示节点和关系，并通过向量相似度搜索来实现图查询功能。

## 项目目标

将 Neo4j 中的图数据库场景（电影、演员、导演关系网络）用向量数据库重新实现，包括：
- 节点和关系的向量化表示
- 相似度搜索实现图查询
- 路径查找和推荐系统
- 社区发现和影响力分析

## 支持的数据库

- **OrientDB** - 多模型图数据库（默认，原生图数据库支持）
- **Qdrant** - 高性能向量数据库，支持复杂查询
- **ChromaDB** - 轻量级向量数据库，易于使用
- **Milvus** - 大规模向量数据库
- **Weaviate** - 功能丰富，支持图查询

## 前置要求

1. **Python 3.8+**

2. **安装依赖**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 快速开始

### 使用 OrientDB（推荐）

OrientDB 是原生图数据库，提供强大的图查询能力。

#### 步骤1: 安装和启动 OrientDB

**方式一：使用 Docker（推荐）**
```bash
docker run -d \
  --name orientdb \
  -p 2424:2424 \
  -p 2480:2480 \
  -e ORIENTDB_ROOT_PASSWORD=root \
  orientdb:latest
```

**方式二：本地安装**
1. 从 [OrientDB 官网](https://orientdb.com/download/) 下载
2. 解压并启动：
   ```bash
   cd orientdb-*/bin
   ./server.sh
   ```

#### 步骤2: 运行代码

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行基础演示
python3 demo.py

# 运行数据挖掘查询
python3 mining_queries.py
```

### 使用其他数据库

#### 使用 Qdrant

```python
from implementations.qdrant_impl import QdrantGraphDB
db = QdrantGraphDB(use_local=True)  # 本地内存模式
```

#### 使用 ChromaDB

```python
from implementations.chroma_impl import ChromaGraphDB
db = ChromaGraphDB(persist_directory="./chroma_db")
```

### 使用其他向量数据库

修改 `demo.py` 中的导入和初始化代码来选择不同的向量数据库实现。

## 项目结构

```
vector-db-learning/
├── README.md              # 项目说明
├── requirements.txt       # Python 依赖
├── demo.py               # 基础演示代码
├── mining_queries.py     # 数据挖掘查询示例
├── vector_utils.py       # 向量工具函数
├── models/               # 数据模型定义
│   └── graph_models.py
└── implementations/      # 不同向量数据库的实现
    ├── chroma_impl.py    # ChromaDB 实现
    ├── qdrant_impl.py    # Qdrant 实现
    └── weaviate_impl.py  # Weaviate 实现
```

## 核心概念

### 1. 节点向量化

将图节点（Person、Movie）转换为向量：
- 使用节点的属性（名称、类型、元数据）生成嵌入
- 存储节点 ID、类型、属性等信息

### 2. 关系向量化

将图关系（ACTED_IN、DIRECTED）转换为向量：
- 使用源节点、目标节点和关系类型生成嵌入
- 支持关系属性（如权重、时间等）

### 3. 查询实现

- **节点查询**：通过向量相似度搜索找到相似节点
- **关系查询**：通过组合节点向量找到关系
- **路径查询**：通过多跳向量搜索实现路径查找
- **推荐系统**：基于向量相似度进行推荐

## 与 Neo4j 的对比

| 功能 | Neo4j | 向量数据库实现 |
|------|-------|---------------|
| 节点存储 | 原生图结构 | 向量嵌入 + 元数据 |
| 关系查询 | Cypher 查询 | 向量相似度搜索 |
| 路径查找 | 图遍历算法 | 多跳向量搜索 |
| 推荐系统 | 基于图算法 | 基于向量相似度 |
| 扩展性 | 适合复杂图查询 | 适合相似度搜索 |

## 使用场景

向量数据库实现图场景的优势：
- ✅ 支持语义搜索（通过文本相似度）
- ✅ 可以结合 LLM 进行智能查询
- ✅ 适合推荐系统和相似度匹配
- ✅ 易于与 AI/ML 模型集成

## 示例查询

### 查找相似演员
```python
# 通过向量相似度找到与 Tom Hanks 相似的演员
similar_actors = find_similar_nodes("Tom Hanks", node_type="Person", top_k=5)
```

### 电影推荐
```python
# 基于用户喜欢的电影，推荐相似电影
recommendations = recommend_movies("The Matrix", top_k=10)
```

### 路径查找
```python
# 找到两个演员之间的连接路径
path = find_path("Tom Hanks", "Keanu Reeves", max_hops=3)
```

## Milvus 配置说明

### Milvus Lite 模式（默认）

- **优点**：无需启动服务，开箱即用
- **适用场景**：开发、测试、小规模数据
- **使用方式**：`MilvusGraphDB(use_lite=True)`

### Milvus 服务模式

- **优点**：性能更好，支持大规模数据
- **适用场景**：生产环境、大规模数据
- **使用方式**：
  1. 启动 Milvus 服务（Docker 或本地安装）
  2. `MilvusGraphDB(host="localhost", port=19530, use_lite=False)`

## 参考资料

- [Milvus 官方文档](https://milvus.io/docs)
- [Milvus Python SDK](https://github.com/milvus-io/pymilvus)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Qdrant 文档](https://qdrant.tech/documentation/)
- [Weaviate 文档](https://weaviate.io/developers/weaviate)
- [向量数据库对比](https://www.pinecone.io/learn/vector-database/)

