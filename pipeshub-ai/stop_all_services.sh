#!/bin/bash

# 停止所有服务的脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "停止 PipesHub AI 所有服务"
echo "=========================================="

# 停止 screen 会话
if command -v screen >/dev/null 2>&1; then
    echo "🛑 停止 screen 会话..."
    screen -ls | grep pipeshub | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit 2>/dev/null || true
fi

# 停止 tmux 会话
if command -v tmux >/dev/null 2>&1; then
    echo "🛑 停止 tmux 会话..."
    tmux ls | grep pipeshub | cut -d: -f1 | xargs -I {} tmux kill-session -t {} 2>/dev/null || true
fi

# 停止通过进程名运行的服务
echo "🛑 停止后台进程..."
pkill -f "app.connectors_main" || true
pkill -f "app.indexing_main" || true
pkill -f "app.query_main" || true
pkill -f "app.docling_main" || true
pkill -f "backend/nodejs/apps" || true
pkill -f "frontend.*vite" || true

sleep 2

echo "✅ 所有服务已停止"
echo ""
echo "查看是否还有残留进程："
ps aux | grep -E "connectors_main|indexing_main|query_main|docling_main|nodejs/apps|frontend" | grep -v grep || echo "没有残留进程"



