#!/bin/bash
# Neo4j Demo 启动脚本

cd "$(dirname "$0")"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "错误: 虚拟环境不存在，请先运行: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 检查参数
case "${1:-demo}" in
    demo)
        echo "运行主 Demo..."
        python3 demo.py
        ;;
    show)
        echo "运行查询展示..."
        python3 show_demo.py
        ;;
    visual)
        echo "运行可视化展示..."
        python3 visual_demo.py
        ;;
    *)
        echo "用法: $0 [demo|show|visual]"
        echo ""
        echo "选项:"
        echo "  demo    - 运行主演示程序（默认）"
        echo "  show    - 运行详细查询结果展示"
        echo "  visual  - 运行可视化图结构展示"
        exit 1
        ;;
esac

