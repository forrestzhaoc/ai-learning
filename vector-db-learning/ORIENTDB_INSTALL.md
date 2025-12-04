# OrientDB 安装指南

## 当前状态
OrientDB 服务未运行。需要先启动 OrientDB 才能使用。

## 安装方法

### 方法1: Docker（推荐，但需要网络）

```bash
docker run -d \
  --name orientdb \
  -p 2424:2424 \
  -p 2480:2480 \
  -e ORIENTDB_ROOT_PASSWORD=root \
  orientdb:latest
```

### 方法2: 手动安装（如果 Docker 无法使用）

1. **下载 OrientDB**:
   ```bash
   cd /tmp
   wget https://orientdb.com/download.php?file=orientdb-community-3.2.20.tar.gz
   tar -xzf orientdb-community-3.2.20.tar.gz
   mv orientdb-community-3.2.20 /opt/orientdb
   ```

2. **启动 OrientDB**:
   ```bash
   cd /opt/orientdb
   bin/server.sh
   ```

3. **验证**:
   - 访问 http://localhost:2480
   - 或检查端口: `netstat -tln | grep 2424`

## 验证安装

运行以下命令检查 OrientDB 是否运行：

```bash
netstat -tln | grep 2424
# 或
ss -tln | grep 2424
```

如果看到端口 2424 在监听，说明 OrientDB 已启动。

## 运行代码

```bash
cd /home/ubuntu/projects/ai-learning/vector-db-learning
source venv/bin/activate
python3 demo.py
```

## 故障排除

如果连接失败：
1. 检查 OrientDB 服务是否运行
2. 检查防火墙设置
3. 确认端口 2424 未被占用
4. 查看 OrientDB 日志
