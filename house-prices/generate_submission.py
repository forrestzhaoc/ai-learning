#!/usr/bin/env python3
"""
自动查找数据文件并生成提交文件
"""
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

def find_data_files():
    """查找数据文件"""
    possible_locations = [
        DATA_DIR / "train.csv",
        DATA_DIR / "test.csv",
        Path.home() / "Downloads" / "train.csv",
        Path.home() / "Downloads" / "test.csv",
        Path("/tmp") / "train.csv",
        Path("/tmp") / "test.csv",
    ]
    
    train_file = None
    test_file = None
    
    # 检查默认位置
    if (DATA_DIR / "train.csv").exists():
        train_file = DATA_DIR / "train.csv"
    if (DATA_DIR / "test.csv").exists():
        test_file = DATA_DIR / "test.csv"
    
    # 检查其他可能的位置
    for loc in possible_locations:
        if loc.name == "train.csv" and loc.exists() and train_file is None:
            train_file = loc
        if loc.name == "test.csv" and loc.exists() and test_file is None:
            test_file = loc
    
    return train_file, test_file

def main():
    print("=" * 60)
    print("House Prices - 生成提交文件")
    print("=" * 60)
    print()
    
    # 查找数据文件
    train_file, test_file = find_data_files()
    
    if train_file is None or test_file is None:
        print("❌ 未找到数据文件")
        print()
        print("请确保以下文件存在：")
        print(f"  - {DATA_DIR / 'train.csv'}")
        print(f"  - {DATA_DIR / 'test.csv'}")
        print()
        print("数据文件可以从以下位置下载：")
        print("  https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
        print()
        print("或者运行下载脚本：")
        print("  python download_data.py")
        print()
        return 1
    
    print(f"✓ 找到训练数据: {train_file}")
    print(f"✓ 找到测试数据: {test_file}")
    print()
    
    # 运行训练脚本
    print("开始训练模型并生成提交文件...")
    print()
    
    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "train.py"),
        "--train", str(train_file),
        "--test", str(test_file),
        "--model-type", "xgb",
        "--cv-folds", "5",
    ]
    
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    
    if result.returncode == 0:
        submission_file = ROOT_DIR / "submissions" / "house_prices_submission.csv"
        print()
        print("=" * 60)
        print("✓ 提交文件生成成功！")
        print("=" * 60)
        print(f"提交文件位置: {submission_file}")
        print()
        if submission_file.exists():
            import pandas as pd
            df = pd.read_csv(submission_file)
            print(f"预测数量: {len(df)}")
            print(f"价格范围: ${df['SalePrice'].min():,.0f} - ${df['SalePrice'].max():,.0f}")
            print(f"平均价格: ${df['SalePrice'].mean():,.0f}")
        return 0
    else:
        print()
        print("❌ 生成提交文件时出现错误")
        return result.returncode

if __name__ == "__main__":
    sys.exit(main())



