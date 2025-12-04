#!/bin/bash

# 启动所有服务的脚本（使用 screen 或 tmux）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "启动 PipesHub AI 所有服务"
echo "=========================================="

# 检查 screen 或 tmux
if command -v screen >/dev/null 2>&1; then
    SESSION_MANAGER="screen"
elif command -v tmux >/dev/null 2>&1; then
    SESSION_MANAGER="tmux"
else
    echo "⚠️  未找到 screen 或 tmux，将使用后台进程启动"
    SESSION_MANAGER="background"
fi

# Python 虚拟环境路径
PYTHON_VENV="$SCRIPT_DIR/backend/python/venv"

# 启动函数
start_service() {
    local name=$1
    local cmd=$2
    local log_file="$SCRIPT_DIR/logs/${name}.log"
    
    mkdir -p "$SCRIPT_DIR/logs"
    
    if [ "$SESSION_MANAGER" = "screen" ]; then
        screen -dmS "pipeshub-$name" bash -c "$cmd 2>&1 | tee $log_file"
        echo "✅ $name 服务已在 screen 会话 'pipeshub-$name' 中启动"
    elif [ "$SESSION_MANAGER" = "tmux" ]; then
        tmux new-session -d -s "pipeshub-$name" "$cmd 2>&1 | tee $log_file"
        echo "✅ $name 服务已在 tmux 会话 'pipeshub-$name' 中启动"
    else
        nohup bash -c "$cmd" > "$log_file" 2>&1 &
        echo "✅ $name 服务已在后台启动 (PID: $!)"
    fi
}

# 激活 Python 环境并启动服务
activate_python() {
    source "$PYTHON_VENV/bin/activate"
}

echo "🚀 启动服务..."

# 启动 Python 连接器服务
start_service "connectors" "cd $SCRIPT_DIR/backend/python && source venv/bin/activate && python -m app.connectors_main"

# 启动 Python 索引服务
start_service "indexing" "cd $SCRIPT_DIR/backend/python && source venv/bin/activate && python -m app.indexing_main"

# 启动 Python 查询服务
start_service "query" "cd $SCRIPT_DIR/backend/python && source venv/bin/activate && python -m app.query_main"

# 启动 Python Docling 服务
start_service "docling" "cd $SCRIPT_DIR/backend/python && source venv/bin/activate && python -m app.docling_main"

# 启动 Node.js 后端服务
start_service "nodejs" "cd $SCRIPT_DIR/backend/nodejs/apps && npm run dev"

# 启动前端服务
start_service "frontend" "cd $SCRIPT_DIR/frontend && npm run dev"

echo ""
echo "=========================================="
echo "✅ 所有服务已启动！"
echo "=========================================="
echo ""
if [ "$SESSION_MANAGER" = "screen" ]; then
    echo "查看服务："
    echo "  screen -ls                    # 列出所有会话"
    echo "  screen -r pipeshub-connectors  # 连接到连接器服务"
    echo "  screen -r pipeshub-indexing   # 连接到索引服务"
    echo "  screen -r pipeshub-query       # 连接到查询服务"
    echo "  screen -r pipeshub-docling     # 连接到 Docling 服务"
    echo "  screen -r pipeshub-nodejs      # 连接到 Node.js 后端"
    echo "  screen -r pipeshub-frontend    # 连接到前端"
elif [ "$SESSION_MANAGER" = "tmux" ]; then
    echo "查看服务："
    echo "  tmux ls                        # 列出所有会话"
    echo "  tmux attach -t pipeshub-connectors  # 连接到连接器服务"
    echo "  tmux attach -t pipeshub-indexing    # 连接到索引服务"
    echo "  tmux attach -t pipeshub-query       # 连接到查询服务"
    echo "  tmux attach -t pipeshub-docling     # 连接到 Docling 服务"
    echo "  tmux attach -t pipeshub-nodejs      # 连接到 Node.js 后端"
    echo "  tmux attach -t pipeshub-frontend    # 连接到前端"
else
    echo "查看日志："
    echo "  tail -f $SCRIPT_DIR/logs/connectors.log"
    echo "  tail -f $SCRIPT_DIR/logs/indexing.log"
    echo "  tail -f $SCRIPT_DIR/logs/query.log"
    echo "  tail -f $SCRIPT_DIR/logs/docling.log"
    echo "  tail -f $SCRIPT_DIR/logs/nodejs.log"
    echo "  tail -f $SCRIPT_DIR/logs/frontend.log"
fi
echo ""
echo "服务访问地址："
echo "  - 前端: http://localhost:3000"
echo "  - Node.js 后端: http://localhost:8000"
echo "  - 查询服务: http://localhost:8000"
echo "  - 连接器服务: http://localhost:8088"
echo "  - 索引服务: http://localhost:8091"
echo "  - Docling 服务: http://localhost:8081"
echo ""
echo "停止所有服务："
echo "  ./stop_all_services.sh"
echo ""



