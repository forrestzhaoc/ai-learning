#!/usr/bin/env python3
"""
Cube API 使用示例
演示如何通过 Python 调用 Cube REST API
"""

import requests
import json
from datetime import datetime

# Cube API 地址
CUBE_API = "http://172.16.0.4:4000/cubejs-api/v1"

def make_query(query):
    """执行 Cube 查询"""
    response = requests.get(
        f"{CUBE_API}/load",
        params={"query": json.dumps(query)}
    )
    return response.json()

def print_section(title):
    """打印分隔标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ============================================================
# 示例 1：基础查询 - 订单总数
# ============================================================
print_section("示例 1: 查询订单总数")

query1 = {
    "measures": ["Orders.count"]
}

result1 = make_query(query1)
count = result1['data'][0]['Orders.count']
print(f"📊 订单总数: {count}")

# ============================================================
# 示例 2：分组查询 - 按产品统计
# ============================================================
print_section("示例 2: 按产品分组统计")

query2 = {
    "measures": ["Orders.count", "Orders.totalAmount", "Orders.averageAmount"],
    "dimensions": ["Orders.product"],
    "order": {"Orders.totalAmount": "desc"}
}

result2 = make_query(query2)
print(f"\n{'产品':<15} {'数量':>8} {'总额':>12} {'平均':>12}")
print("-" * 50)
for row in result2['data']:
    product = row['Orders.product']
    count = row['Orders.count']
    total = row['Orders.totalAmount']
    avg = row['Orders.averageAmount']
    print(f"{product:<15} {count:>8} ${total:>11.2f} ${avg:>11.2f}")

# ============================================================
# 示例 3：过滤查询 - 只查询特定产品
# ============================================================
print_section("示例 3: 过滤查询（只看 Product A）")

query3 = {
    "measures": ["Orders.count", "Orders.totalAmount"],
    "filters": [{
        "member": "Orders.product",
        "operator": "equals",
        "values": ["Product A"]
    }]
}

result3 = make_query(query3)
data3 = result3['data'][0]
print(f"Product A 订单数: {data3['Orders.count']}")
print(f"Product A 总销售额: ${data3['Orders.totalAmount']}")

# ============================================================
# 示例 4：时间维度查询 - 按日期查看
# ============================================================
print_section("示例 4: 按时间查询")

query4 = {
    "measures": ["Orders.count", "Orders.totalAmount"],
    "timeDimensions": [{
        "dimension": "Orders.createdAt",
        "granularity": "day"
    }],
    "order": {"Orders.createdAt": "asc"}
}

result4 = make_query(query4)
print(f"\n{'日期':<12} {'订单数':>8} {'销售额':>12}")
print("-" * 35)
for row in result4['data']:
    date = row['Orders.createdAt'][:10]  # 只取日期部分
    count = row['Orders.count']
    amount = row['Orders.totalAmount']
    print(f"{date:<12} {count:>8} ${amount:>11.2f}")

# ============================================================
# 示例 5：复杂查询 - 组合多个条件
# ============================================================
print_section("示例 5: 复杂查询（时间范围 + 产品筛选）")

query5 = {
    "measures": ["Orders.count", "Orders.averageAmount"],
    "dimensions": ["Orders.product"],
    "filters": [{
        "member": "Orders.amount",
        "operator": "gte",
        "values": ["150"]  # 金额 >= 150
    }],
    "timeDimensions": [{
        "dimension": "Orders.createdAt",
        "dateRange": ["2024-01-01", "2024-01-05"]
    }]
}

result5 = make_query(query5)
if 'data' in result5 and result5['data']:
    print("\n高价值订单（金额 >= $150）:")
    for row in result5['data']:
        product = row['Orders.product']
        count = row['Orders.count']
        avg = row['Orders.averageAmount']
        print(f"  • {product}: {count} 个订单, 平均 ${avg:.2f}")
else:
    print(f"\n⚠️  查询无结果或出错: {result5.get('error', '未知错误')}")

# ============================================================
# 示例 6：获取原始数据 - 详细订单列表
# ============================================================
print_section("示例 6: 获取详细订单列表")

query6 = {
    "dimensions": [
        "Orders.id",
        "Orders.product",
        "Orders.createdAt"
    ],
    "measures": ["Orders.totalAmount"],
    "order": {"Orders.id": "asc"}
}

result6 = make_query(query6)
print(f"\n{'ID':>4} {'产品':<15} {'日期':<12} {'金额':>10}")
print("-" * 45)
for row in result6['data']:
    order_id = row['Orders.id']
    product = row['Orders.product']
    date = row['Orders.createdAt'][:10]
    amount = row['Orders.totalAmount']
    print(f"{order_id:>4} {product:<15} {date:<12} ${amount:>9.2f}")

# ============================================================
# 总结
# ============================================================
print_section("✅ 所有示例执行完毕")
print("""
这些示例展示了 Cube 的核心功能：
1. ✓ 聚合查询（COUNT, SUM, AVG）
2. ✓ 分组统计（GROUP BY）
3. ✓ 条件过滤（WHERE）
4. ✓ 时间维度分析
5. ✓ 复杂组合查询
6. ✓ 原始数据获取

更多用法请查看：
• Playground: http://172.16.0.4:4000
• 文档: /home/ubuntu/projects/ai-learning/cube-demo/USAGE_GUIDE.md
""")

