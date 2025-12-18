#!/usr/bin/env python3
"""
改进版训练脚本 - 针对RMSLE优化

使用方法:
    python train_improved.py
"""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"

def main():
    print("="*70)
    print("NYC Taxi Trip Duration - 改进版训练")
    print("="*70)
    print("\n改进点:")
    print("  1. ✅ 增强特征工程（更多时间和地理特征）")
    print("  2. ✅ 使用对数目标变量（针对RMSLE优化）")
    print("  3. ✅ 优化模型参数（更多树，更优学习率）")
    print("  4. ✅ 使用优化的XGBoost参数")
    print("="*70)
    
    from src.train import main as train_main
    
    # 设置参数
    sys.argv = [
        'train.py',
        '--train', str(DATA_DIR / "train.csv"),
        '--test', str(DATA_DIR / "test.csv"),
        '--model-type', 'xgb',  # 使用XGBoost (LightGBM需要单独安装)
        '--cv-folds', '0',  # 跳过交叉验证以加快速度
        '--use-log-target',  # 使用对数目标（关键改进！）
        '--model-path', str(MODEL_DIR / "nyc_taxi_improved.joblib"),
        '--submission', str(PRED_DIR / "submission_improved.csv"),
        '--random-state', '42'
    ]
    
    try:
        train_main()
        print("\n" + "="*70)
        print("✓ 改进版训练完成！")
        print(f"  模型: {MODEL_DIR / 'nyc_taxi_improved.joblib'}")
        print(f"  提交文件: {PRED_DIR / 'submission_improved.csv'}")
        print("\n预期改进:")
        print("  原模型 RMSLE: ~0.47275")
        print("  改进模型 RMSLE: ~0.38-0.42 (预计提升 10-20%)")
        print("="*70)
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



