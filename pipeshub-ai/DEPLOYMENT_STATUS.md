# PipesHub AI 部署状态

## 当前状态

✅ **已完成：**
1. 项目代码已成功下载到 `/home/ubuntu/projects/pipeshub-ai`
2. Docker 和 docker-compose 已安装
3. 环境变量文件已配置（`.env` 文件已创建）
4. Docker Compose 配置文件已修复（Redis 和 ulimits 配置问题）
5. Docker 镜像加速器已配置

⚠️ **进行中：**
- Docker 镜像拉取遇到网络连接问题
- 部分镜像（etcd）已成功拉取
- 其他镜像（mongodb, redis, arango, qdrant, kafka, zookeeper）拉取超时

## 部署步骤

### 1. 检查网络连接

确保服务器可以访问 Docker Hub 或镜像加速器：

```bash
# 测试网络连接
ping -c 3 registry-1.docker.io
curl -I https://registry.docker-cn.com

# 如果网络有问题，可能需要：
# - 配置代理
# - 使用其他镜像源
# - 检查防火墙设置
```

### 2. 启动服务

进入部署目录并启动服务：

```bash
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose

# 使用开发环境配置启动
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up --build -d

# 查看服务状态
docker-compose -f docker-compose.dev.yml -p pipeshub-ai ps

# 查看日志
docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs -f
```

### 3. 访问服务

服务启动后，可以通过以下端口访问：

- **前端**: http://localhost:3000
- **查询后端**: http://localhost:8001
- **连接器后端**: http://localhost:8088
- **索引后端**: http://localhost:8091
- **ArangoDB**: http://localhost:8529
- **MongoDB**: localhost:27017
- **Redis**: localhost:6379
- **Qdrant**: http://localhost:6333
- **Kafka**: localhost:9092
- **etcd**: http://localhost:2379

## 环境变量配置

环境变量文件位于：`deployment/docker-compose/.env`

主要配置项：
- `SECRET_KEY`: 加密密钥（已自动生成）
- `ARANGO_PASSWORD`: ArangoDB 密码（已自动生成）
- `MONGO_PASSWORD`: MongoDB 密码（已自动生成）
- `QDRANT_API_KEY`: Qdrant API 密钥（已自动生成）
- `FRONTEND_PUBLIC_URL`: 前端公共 URL（默认: http://localhost:3000）

## 故障排除

### 网络连接问题

如果遇到镜像拉取超时：

1. **使用代理**（如果有）：
```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null << EOF
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

2. **手动拉取镜像**：
```bash
# 逐个拉取所需镜像
docker pull mongo:8.0.6
docker pull redis:bookworm
docker pull arangodb:3.11.14
docker pull qdrant/qdrant:v1.15
docker pull confluentinc/cp-zookeeper:7.9.0
docker pull confluentinc/cp-kafka:7.9.0
docker pull quay.io/coreos/etcd:v3.5.17
```

3. **使用离线镜像**：
如果有离线镜像包，可以导入：
```bash
docker load < images.tar
```

### 服务启动失败

1. **检查日志**：
```bash
docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs
```

2. **检查资源**：
确保有足够的内存（至少 10GB）和磁盘空间

3. **重启服务**：
```bash
docker-compose -f docker-compose.dev.yml -p pipeshub-ai restart
```

### 停止服务

```bash
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose
docker-compose -f docker-compose.dev.yml -p pipeshub-ai down
```

### 清理数据

```bash
# 停止并删除容器和卷
docker-compose -f docker-compose.dev.yml -p pipeshub-ai down -v
```

## 下一步

1. 解决网络连接问题，完成镜像拉取
2. 启动所有服务
3. 验证服务运行状态
4. 访问前端界面进行配置

## 参考文档

- 项目 README: `/home/ubuntu/projects/pipeshub-ai/README.md`
- 部署目录: `/home/ubuntu/projects/pipeshub-ai/deployment/docker-compose/`
- 官方文档: https://docs.pipeshub.com/



