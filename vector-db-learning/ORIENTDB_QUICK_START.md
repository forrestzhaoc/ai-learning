# OrientDB 快速开始

## 1. 启动 OrientDB 服务

### 使用 Docker（推荐）

```bash
docker run -d \
  --name orientdb \
  -p 2424:2424 \
  -p 2480:2480 \
  -e ORIENTDB_ROOT_PASSWORD=root \
  orientdb:latest
```

### 验证服务运行

```bash
# 检查容器状态
docker ps | grep orientdb

# 访问 Web 界面
# http://localhost:2480
# 用户名: root
# 密码: root
```

## 2. 运行代码

```bash
cd /home/ubuntu/projects/ai-learning/vector-db-learning
source venv/bin/activate
python3 demo.py
```

## 3. 如果 OrientDB 未运行

代码会自动检测连接失败，你可以：
- 启动 OrientDB 服务后重试
- 或切换到其他数据库实现（Qdrant、ChromaDB 等）

## OrientDB 的优势

- ✅ 原生图数据库，支持复杂图查询
- ✅ 高性能图遍历
- ✅ 支持 SQL 类似语法
- ✅ 向量嵌入作为节点属性存储
- ✅ 支持事务和 ACID

