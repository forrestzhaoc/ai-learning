#!/bin/bash

# PipesHub AI 部署脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/deployment/docker-compose"

echo "=========================================="
echo "PipesHub AI 部署脚本"
echo "=========================================="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose 未安装，请先安装 docker-compose"
    exit 1
fi

# 进入部署目录
cd "$DEPLOY_DIR"

# 检查环境变量文件
if [ ! -f .env ]; then
    echo "📝 创建环境变量文件..."
    if [ -f env.template ]; then
        cp env.template .env
        # 生成随机密码
        python3 << 'PYEOF'
import secrets
import re

with open('.env', 'r') as f:
    content = f.read()

# 替换密码
content = re.sub(r'SECRET_KEY=.*', f'SECRET_KEY={secrets.token_urlsafe(32)}', content)
content = re.sub(r'ARANGO_PASSWORD=.*', f'ARANGO_PASSWORD={secrets.token_urlsafe(16)}', content)
content = re.sub(r'MONGO_PASSWORD=.*', f'MONGO_PASSWORD={secrets.token_urlsafe(16)}', content)
content = re.sub(r'QDRANT_API_KEY=.*', f'QDRANT_API_KEY={secrets.token_urlsafe(24)}', content)

with open('.env', 'w') as f:
    f.write(content)
PYEOF
        echo "✅ 环境变量文件已创建"
    else
        echo "❌ 找不到 env.template 文件"
        exit 1
    fi
fi

# 检查网络连接
echo "🔍 检查网络连接..."
if ! ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
    echo "⚠️  警告: 网络连接可能有问题"
fi

# 拉取镜像
echo "📥 拉取 Docker 镜像..."
echo "这可能需要一些时间，请耐心等待..."

# 尝试拉取镜像，如果失败则继续
docker-compose -f docker-compose.dev.yml pull || {
    echo "⚠️  部分镜像拉取失败，将尝试启动现有镜像"
}

# 构建和启动服务
echo "🚀 构建并启动服务..."
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up --build -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "📊 服务状态："
docker-compose -f docker-compose.dev.yml -p pipeshub-ai ps

echo ""
echo "=========================================="
echo "✅ 部署完成！"
echo "=========================================="
echo ""
echo "服务访问地址："
echo "  - 前端: http://localhost:3000"
echo "  - 查询后端: http://localhost:8001"
echo "  - 连接器后端: http://localhost:8088"
echo "  - 索引后端: http://localhost:8091"
echo "  - ArangoDB: http://localhost:8529"
echo "  - Qdrant: http://localhost:6333"
echo ""
echo "查看日志: docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs -f"
echo "停止服务: docker-compose -f docker-compose.dev.yml -p pipeshub-ai down"
echo ""



