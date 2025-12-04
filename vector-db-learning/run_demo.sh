#!/bin/bash

# 向量数据库演示脚本

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "正在创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
if [ ! -f "venv/.installed" ]; then
    echo "正在安装依赖..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# 运行演示
case "$1" in
    "mining"|"query")
        echo "运行数据挖掘查询..."
        python3 mining_queries.py
        ;;
    "demo"|"")
        echo "运行基础演示..."
        python3 demo.py
        ;;
    *)
        echo "用法: $0 [demo|mining]"
        echo "  demo   - 运行基础演示（默认）"
        echo "  mining - 运行数据挖掘查询"
        exit 1
        ;;
esac



