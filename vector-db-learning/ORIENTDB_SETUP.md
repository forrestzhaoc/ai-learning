# OrientDB 安装和配置指南

## 快速安装（Docker）

```bash
# 启动 OrientDB 容器
docker run -d \
  --name orientdb \
  -p 2424:2424 \
  -p 2480:2480 \
  -e ORIENTDB_ROOT_PASSWORD=root \
  orientdb:latest
```

## 验证安装

访问 OrientDB Studio（Web 界面）：
- URL: http://localhost:2480
- 默认用户名: root
- 默认密码: root（或你设置的密码）

## 连接信息

- **主机**: localhost
- **端口**: 2424（二进制协议）
- **HTTP 端口**: 2480（Web 界面）
- **默认数据库**: graph_db（会自动创建）
- **默认用户名**: admin
- **默认密码**: admin

## 使用 Python 客户端

```python
from implementations.orientdb_impl import OrientDBGraphDB

# 连接到 OrientDB
db = OrientDBGraphDB(
    host="localhost",
    port=2424,
    database="graph_db",
    username="admin",
    password="admin"
)
```

## 常见问题

### Q: 连接失败怎么办？

A: 确保 OrientDB 服务正在运行：
```bash
# 检查 Docker 容器
docker ps | grep orientdb

# 查看日志
docker logs orientdb
```

### Q: 如何重置数据库？

A: 在 OrientDB Studio 中删除数据库，或使用命令行：
```sql
DROP DATABASE graph_db
```

### Q: 如何备份数据？

A: 使用 OrientDB 的备份功能：
```bash
docker exec orientdb /orientdb/bin/console.sh "CONNECT remote:localhost/graph_db admin admin; EXPORT DATABASE /tmp/backup.json"
```

## OrientDB 的优势

- ✅ 原生图数据库，支持复杂的图查询
- ✅ 支持 SQL 类似语法（OrientDB SQL）
- ✅ 高性能图遍历
- ✅ 支持事务
- ✅ 支持多种数据模型（图、文档、键值）

## 参考资源

- [OrientDB 官方文档](https://orientdb.com/docs/)
- [OrientDB Python 客户端](https://github.com/orientechnologies/pyorient)



