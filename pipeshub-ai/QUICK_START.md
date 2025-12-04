# PipesHub AI 快速启动指南

## 📋 部署状态总结

### ✅ 已完成的工作

1. **项目代码下载**
   - 项目已下载到: `/home/ubuntu/projects/pipeshub-ai`
   - 所有源代码文件完整

2. **环境配置**
   - Docker 已安装并运行
   - docker-compose 已安装 (版本 1.29.2)
   - 环境变量文件已创建: `deployment/docker-compose/.env`
   - 所有密码和密钥已自动生成

3. **配置文件修复**
   - Redis 配置问题已修复
   - Docker Compose ulimits 配置已修复
   - Docker 镜像加速器已配置

### ⚠️ 当前问题

**网络连接问题**: Docker 镜像拉取遇到超时，无法从 Docker Hub 或镜像加速器下载镜像。

**影响**: 服务无法启动，因为缺少必要的 Docker 镜像。

## 🚀 解决方案

### 方案 1: 配置网络代理（推荐）

如果您的服务器需要通过代理访问外网：

```bash
# 创建 Docker 代理配置
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null << EOF
[Service]
Environment="HTTP_PROXY=http://your-proxy:port"
Environment="HTTPS_PROXY=http://your-proxy:port"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF

# 重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 然后重新启动部署
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up --build -d
```

### 方案 2: 手动拉取镜像

在网络条件较好的时候，手动拉取所需镜像：

```bash
# 所需镜像列表
docker pull mongo:8.0.6
docker pull redis:bookworm
docker pull arangodb:3.11.14
docker pull qdrant/qdrant:v1.15
docker pull confluentinc/cp-zookeeper:7.9.0
docker pull confluentinc/cp-kafka:7.9.0
docker pull quay.io/coreos/etcd:v3.5.17

# 拉取完成后启动服务
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up --build -d
```

### 方案 3: 使用离线镜像包

如果有离线镜像包，可以导入：

```bash
# 导入镜像
docker load < pipeshub-images.tar

# 启动服务
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up -d
```

### 方案 4: 使用部署脚本

使用提供的部署脚本：

```bash
cd /home/ubuntu/projects/pipeshub-ai
./deploy.sh
```

## 📝 启动服务

一旦镜像问题解决，使用以下命令启动服务：

```bash
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose

# 启动服务（开发环境）
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up --build -d

# 查看服务状态
docker-compose -f docker-compose.dev.yml -p pipeshub-ai ps

# 查看日志
docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs -f

# 停止服务
docker-compose -f docker-compose.dev.yml -p pipeshub-ai down
```

## 🌐 服务访问地址

服务启动后，可以通过以下地址访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端 | http://localhost:3000 | Web 界面 |
| 查询后端 | http://localhost:8001 | API 服务 |
| 连接器后端 | http://localhost:8088 | 连接器服务 |
| 索引后端 | http://localhost:8091 | 索引服务 |
| ArangoDB | http://localhost:8529 | 图数据库 |
| Qdrant | http://localhost:6333 | 向量数据库 |
| MongoDB | localhost:27017 | 文档数据库 |
| Redis | localhost:6379 | 缓存服务 |
| Kafka | localhost:9092 | 消息队列 |
| etcd | http://localhost:2379 | 配置存储 |

## 🔧 环境变量

环境变量文件位置: `deployment/docker-compose/.env`

主要配置项（已自动生成）:
- `SECRET_KEY`: 加密密钥
- `ARANGO_PASSWORD`: ArangoDB 密码
- `MONGO_PASSWORD`: MongoDB 密码  
- `QDRANT_API_KEY`: Qdrant API 密钥

## 📚 相关文件

- 部署状态: `DEPLOYMENT_STATUS.md`
- 部署脚本: `deploy.sh`
- 环境变量模板: `deployment/docker-compose/env.template`
- Docker Compose 配置: `deployment/docker-compose/docker-compose.dev.yml`

## 🆘 故障排除

### 检查服务状态

```bash
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose
docker-compose -f docker-compose.dev.yml -p pipeshub-ai ps
```

### 查看服务日志

```bash
# 查看所有服务日志
docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs

# 查看特定服务日志
docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs pipeshub-ai
docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs mongodb
```

### 重启服务

```bash
docker-compose -f docker-compose.dev.yml -p pipeshub-ai restart
```

### 清理并重新部署

```bash
# 停止并删除所有容器和卷
docker-compose -f docker-compose.dev.yml -p pipeshub-ai down -v

# 重新启动
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up --build -d
```

## 📖 更多信息

- 项目 README: `README.md`
- 官方文档: https://docs.pipeshub.com/
- GitHub: https://github.com/forrestzhaoc/pipeshub-ai



