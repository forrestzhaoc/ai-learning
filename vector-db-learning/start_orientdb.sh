#!/bin/bash
# OrientDB 启动脚本

echo "正在尝试启动 OrientDB..."

# 方法1: 检查是否已有容器
if docker ps -a | grep -q orientdb; then
    echo "发现现有 OrientDB 容器，正在启动..."
    docker start orientdb
    sleep 5
    if docker ps | grep -q orientdb; then
        echo "✓ OrientDB 已启动"
        exit 0
    fi
fi

# 方法2: 尝试创建新容器
echo "正在创建新的 OrientDB 容器..."
docker run -d \
  --name orientdb \
  -p 2424:2424 \
  -p 2480:2480 \
  -e ORIENTDB_ROOT_PASSWORD=root \
  orientdb:latest 2>&1

sleep 5

if docker ps | grep -q orientdb; then
    echo "✓ OrientDB 已成功启动"
    echo "  - 二进制端口: 2424"
    echo "  - Web 界面: http://localhost:2480"
    echo "  - 用户名: root"
    echo "  - 密码: root"
else
    echo "⚠ OrientDB 启动失败"
    echo "请检查 Docker 是否正常运行，或手动安装 OrientDB"
    exit 1
fi
