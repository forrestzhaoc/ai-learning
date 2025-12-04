#!/bin/bash

# 网络问题修复和启动脚本

set -e

echo "=========================================="
echo "PipesHub AI 网络修复和启动脚本"
echo "=========================================="

# 检查网络连接
echo "🔍 检查网络连接..."
if ! ping -c 2 -W 2 8.8.8.8 &> /dev/null; then
    echo "❌ 网络连接异常，请检查网络配置"
    exit 1
fi

# 尝试多个镜像源
echo "📥 尝试从多个镜像源拉取镜像..."

MIRRORS=(
    "https://docker.nju.edu.cn"
    "https://docker.mirrors.sjtug.sjtu.edu.cn"
    "https://dockerproxy.com"
    "https://docker.mirrors.ustc.edu.cn"
)

# 更新 Docker 配置
echo "⚙️  配置 Docker 镜像加速器..."
sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
  "registry-mirrors": [
    "https://docker.nju.edu.cn",
    "https://docker.mirrors.sjtug.sjtu.edu.cn",
    "https://dockerproxy.com",
    "https://docker.mirrors.ustc.edu.cn"
  ],
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
sleep 5

echo "✅ Docker 已重启"

# 进入部署目录
cd /home/ubuntu/projects/pipeshub-ai/deployment/docker-compose

# 尝试拉取镜像（增加超时时间）
echo "📥 开始拉取 Docker 镜像..."
echo "这可能需要较长时间，请耐心等待..."

# 设置超时时间
export DOCKER_CLIENT_TIMEOUT=1200
export COMPOSE_HTTP_TIMEOUT=1200

# 尝试拉取镜像
docker-compose -f docker-compose.dev.yml pull --ignore-pull-failures || {
    echo "⚠️  部分镜像拉取失败，继续尝试启动..."
}

# 检查已拉取的镜像
echo ""
echo "📊 已拉取的镜像："
docker images | grep -E "mongo|redis|arango|qdrant|kafka|zookeeper|etcd|pipeshub" || echo "暂无相关镜像"

# 尝试启动服务
echo ""
echo "🚀 尝试启动服务..."
docker-compose -f docker-compose.dev.yml -p pipeshub-ai up -d --remove-orphans 2>&1 | tail -20

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 15

# 检查服务状态
echo ""
echo "📊 服务状态："
docker-compose -f docker-compose.dev.yml -p pipeshub-ai ps

echo ""
echo "=========================================="
echo "✅ 启动完成！"
echo "=========================================="
echo ""
echo "如果服务未完全启动，可能是网络问题导致镜像未拉取成功。"
echo "请检查网络连接或配置代理后重试。"
echo ""
echo "查看日志: docker-compose -f docker-compose.dev.yml -p pipeshub-ai logs -f"
echo ""



