# Neo4j Demo 使用指南

## 快速开始

### 1. 启动 Neo4j 服务

```bash
# 检查 Neo4j 状态
sudo systemctl status neo4j

# 如果未运行，启动服务
sudo systemctl start neo4j
```

### 2. 运行 Demo

**最简单的方式：**
```bash
./run_demo.sh
```

**或者指定运行模式：**
```bash
./run_demo.sh demo      # 主演示程序
./run_demo.sh show      # 详细查询展示
./run_demo.sh visual    # 可视化图结构
```

### 3. 查看效果

- **命令行查看**：运行上述脚本即可看到结果
- **浏览器查看**：访问 http://localhost:7474
  - 用户名: `neo4j`
  - 密码: `password`

## 常见问题

### Q: 提示找不到 python 命令？
A: 使用 `python3` 而不是 `python`，或者使用便捷脚本 `./run_demo.sh`

### Q: 无法连接到数据库？
A: 检查 Neo4j 服务是否运行：
```bash
sudo systemctl status neo4j
sudo systemctl start neo4j
```

### Q: 无法访问 http://localhost:7474？
A: 确保 Neo4j 配置为监听所有接口：
```bash
# 检查配置
sudo grep "server.default_listen_address" /etc/neo4j/neo4j.conf

# 如果显示为注释状态，取消注释并重启
sudo sed -i 's/^#server.default_listen_address=0.0.0.0/server.default_listen_address=0.0.0.0/' /etc/neo4j/neo4j.conf
sudo systemctl restart neo4j
```

## 文件说明

- `demo.py` - 主演示程序，展示完整的 CRUD 操作
- `show_demo.py` - 详细查询结果表格展示
- `visual_demo.py` - ASCII 图形可视化展示
- `run_demo.sh` - 便捷启动脚本（自动处理虚拟环境）
- `cypher_examples.cypher` - 29 个 Cypher 查询示例

## 下一步

1. 查看 `cypher_examples.cypher` 学习更多查询
2. 在 Neo4j Browser 中尝试不同的查询
3. 修改 `demo.py` 添加自己的数据
