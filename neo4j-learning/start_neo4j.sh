#!/bin/bash
# Neo4j 启动脚本

echo "正在检查 Neo4j 数据库..."

# 检查 Docker 是否安装
if command -v docker &> /dev/null; then
    echo "检测到 Docker，使用 Docker 启动 Neo4j..."
    
    # 检查容器是否已存在
    if docker ps -a | grep -q neo4j; then
        echo "Neo4j 容器已存在，正在启动..."
        docker start neo4j
    else
        echo "创建并启动新的 Neo4j 容器..."
        docker run -d \
            --name neo4j \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/password \
            -e NEO4J_PLUGINS='["apoc"]' \
            neo4j:latest
    fi
    
    echo ""
    echo "Neo4j 正在启动中..."
    echo "等待数据库就绪（大约 10-20 秒）..."
    sleep 15
    
    echo ""
    echo "✓ Neo4j 数据库已启动！"
    echo ""
    echo "访问信息："
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - Bolt 端口: localhost:7687"
    echo "  - 用户名: neo4j"
    echo "  - 密码: password"
    echo ""
    echo "运行 demo:"
    echo "  source venv/bin/activate && python demo.py"
    
elif command -v neo4j &> /dev/null; then
    echo "检测到本地 Neo4j 安装，正在启动..."
    sudo neo4j start
    echo ""
    echo "✓ Neo4j 数据库已启动！"
    echo "访问 http://localhost:7474 查看 Neo4j Browser"
    
else
    echo "错误：未找到 Docker 或 Neo4j 安装"
    echo ""
    echo "请选择以下方式之一安装 Neo4j："
    echo ""
    echo "方式 1: 使用 Docker（推荐）"
    echo "  sudo apt install docker.io"
    echo "  sudo systemctl start docker"
    echo "  然后重新运行此脚本"
    echo ""
    echo "方式 2: 安装 Neo4j Community Edition"
    echo "  wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -"
    echo "  echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list"
    echo "  sudo apt update"
    echo "  sudo apt install neo4j"
    echo "  sudo systemctl enable neo4j"
    echo "  sudo systemctl start neo4j"
    exit 1
fi

