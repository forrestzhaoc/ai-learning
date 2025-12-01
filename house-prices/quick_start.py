#!/usr/bin/env python3
"""
House Prices 快速开始脚本

这是一个简单的示例脚本，展示如何使用训练脚本。
确保 data/ 目录下有 train.csv 和 test.csv 文件。
"""
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def main():
    train_file = ROOT_DIR / "data" / "train.csv"
    test_file = ROOT_DIR / "data" / "test.csv"
    
    # 检查数据文件是否存在
    if not train_file.exists():
        print(f"错误: 找不到训练数据文件 {train_file}")
        print("请从 Kaggle 下载 train.csv 并放到 data/ 目录下")
        sys.exit(1)
    
    if not test_file.exists():
        print(f"错误: 找不到测试数据文件 {test_file}")
        print("请从 Kaggle 下载 test.csv 并放到 data/ 目录下")
        sys.exit(1)
    
    print("开始训练 House Prices 模型...")
    print("=" * 60)
    
    # 运行训练脚本
    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "train.py"),
        "--model-type", "xgb",
        "--cv-folds", "5",
    ]
    
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"提交文件位置: {ROOT_DIR / 'submissions' / 'house_prices_submission.csv'}")
    else:
        print("\n训练过程中出现错误")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()

