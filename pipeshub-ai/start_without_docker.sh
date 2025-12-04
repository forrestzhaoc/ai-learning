#!/bin/bash

# PipesHub AI 不使用 Docker 的启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "PipesHub AI - 不使用 Docker 启动"
echo "=========================================="

# 检查基本工具
echo "🔍 检查基本工具..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 未安装"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js 未安装"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm 未安装"; exit 1; }

echo "✅ 基本工具检查完成"

# 安装数据库服务（如果未安装）
echo ""
echo "📦 检查并安装数据库服务..."

# 安装 Redis
if ! command -v redis-server >/dev/null 2>&1; then
    echo "📥 安装 Redis..."
    sudo apt-get update
    sudo apt-get install -y redis-server
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
    echo "✅ Redis 已安装并启动"
else
    echo "✅ Redis 已安装"
    sudo systemctl start redis-server || true
fi

# 安装 MongoDB
if ! command -v mongod >/dev/null 2>&1; then
    echo "📥 安装 MongoDB..."
    curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
    sudo apt-get update
    sudo apt-get install -y mongodb-org
    sudo systemctl enable mongod
    sudo systemctl start mongod
    echo "✅ MongoDB 已安装并启动"
else
    echo "✅ MongoDB 已安装"
    sudo systemctl start mongod || true
fi

# 注意：ArangoDB, Qdrant, Kafka, etcd 需要单独安装或使用 Docker
# 为了简化，我们先启动已有的服务

echo ""
echo "⚠️  注意：以下服务需要单独安装或使用 Docker 运行："
echo "   - ArangoDB (图数据库)"
echo "   - Qdrant (向量数据库)"
echo "   - Kafka + Zookeeper (消息队列)"
echo "   - etcd (配置存储)"
echo ""
echo "如果这些服务未安装，可以使用 Docker 单独运行它们："
echo "   docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD=your_password arangodb/arangodb:latest"
echo "   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant"
echo "   docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.17"
echo ""

# 设置 Python 后端环境
echo "🐍 设置 Python 后端环境..."
cd "$SCRIPT_DIR/backend/python"

if [ ! -d "venv" ]; then
    echo "📦 创建 Python 虚拟环境..."
    python3 -m venv venv
fi

echo "📦 激活虚拟环境并安装依赖..."
source venv/bin/activate

# 检查是否已安装依赖
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 安装 Python 依赖（这可能需要一些时间）..."
    pip install --upgrade pip
    pip install -e .
    echo "📥 安装语言模型..."
    python -m spacy download en_core_web_sm || echo "⚠️  spacy 模型下载失败，继续..."
    python -c "import nltk; nltk.download('punkt')" || echo "⚠️  nltk 数据下载失败，继续..."
else
    echo "✅ Python 依赖已安装"
fi

# 创建环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建 Python 后端环境变量文件..."
    if [ -f "../env.template" ]; then
        cp ../env.template .env
        echo "✅ 环境变量文件已创建，请根据需要修改 .env 文件"
    fi
fi

# 设置 Node.js 后端环境
echo ""
echo "📦 设置 Node.js 后端环境..."
cd "$SCRIPT_DIR/backend/nodejs/apps"

if [ ! -d "node_modules" ]; then
    echo "📥 安装 Node.js 依赖..."
    npm install
else
    echo "✅ Node.js 依赖已安装"
fi

# 创建环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建 Node.js 后端环境变量文件..."
    if [ -f "../../env.template" ]; then
        cp ../../env.template .env
        echo "✅ 环境变量文件已创建，请根据需要修改 .env 文件"
    fi
fi

# 设置前端环境
echo ""
echo "⚛️  设置前端环境..."
cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    npm install || yarn install
else
    echo "✅ 前端依赖已安装"
fi

# 创建环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建前端环境变量文件..."
    if [ -f "env.template" ]; then
        cp env.template .env
        echo "✅ 环境变量文件已创建，请根据需要修改 .env 文件"
    fi
fi

echo ""
echo "=========================================="
echo "✅ 环境设置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 确保所有数据库服务正在运行（MongoDB, Redis, ArangoDB, Qdrant, Kafka, etcd）"
echo "2. 根据需要修改环境变量文件："
echo "   - backend/python/.env"
echo "   - backend/nodejs/apps/.env"
echo "   - frontend/.env"
echo ""
echo "3. 启动服务（需要在不同的终端窗口中运行）："
echo ""
echo "   # 终端 1: Python 连接器服务"
echo "   cd $SCRIPT_DIR/backend/python"
echo "   source venv/bin/activate"
echo "   python -m app.connectors_main"
echo ""
echo "   # 终端 2: Python 索引服务"
echo "   cd $SCRIPT_DIR/backend/python"
echo "   source venv/bin/activate"
echo "   python -m app.indexing_main"
echo ""
echo "   # 终端 3: Python 查询服务"
echo "   cd $SCRIPT_DIR/backend/python"
echo "   source venv/bin/activate"
echo "   python -m app.query_main"
echo ""
echo "   # 终端 4: Python Docling 服务"
echo "   cd $SCRIPT_DIR/backend/python"
echo "   source venv/bin/activate"
echo "   python -m app.docling_main"
echo ""
echo "   # 终端 5: Node.js 后端服务"
echo "   cd $SCRIPT_DIR/backend/nodejs/apps"
echo "   npm run dev"
echo ""
echo "   # 终端 6: 前端服务"
echo "   cd $SCRIPT_DIR/frontend"
echo "   npm run dev"
echo ""
echo "或者使用提供的启动脚本："
echo "   ./start_all_services.sh"
echo ""



