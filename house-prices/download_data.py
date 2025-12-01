#!/usr/bin/env python3
"""
下载 House Prices 数据文件
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

def download_with_kaggle_api():
    """使用 Kaggle API 下载数据"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("正在从 Kaggle 下载数据...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 下载 House Prices 竞赛数据
        api.competition_download_files(
            'house-prices-advanced-regression-techniques',
            path=str(DATA_DIR),
            unzip=True
        )
        
        print(f"数据已下载到: {DATA_DIR}")
        return True
    except Exception as e:
        print(f"使用 Kaggle API 下载失败: {e}")
        return False

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_file = DATA_DIR / "train.csv"
    test_file = DATA_DIR / "test.csv"
    
    if train_file.exists() and test_file.exists():
        print("数据文件已存在，无需下载")
        return
    
    print("=" * 60)
    print("House Prices 数据下载")
    print("=" * 60)
    print()
    
    # 尝试使用 Kaggle API
    if download_with_kaggle_api():
        print("下载成功！")
        return
    
    # 如果 API 不可用，提供手动下载说明
    print()
    print("=" * 60)
    print("无法自动下载数据")
    print("=" * 60)
    print()
    print("请手动下载数据文件：")
    print("1. 访问: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data")
    print("2. 下载 train.csv 和 test.csv")
    print(f"3. 将文件放到: {DATA_DIR}")
    print()
    print("或者安装 Kaggle API:")
    print("  pip install kaggle")
    print("  # 然后配置 ~/.kaggle/kaggle.json")
    print()

if __name__ == "__main__":
    main()

