"""
NYC Taxi Trip Duration 快速启动脚本

用法:
    python quick_start.py
"""
from pathlib import Path
import sys

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
PRED_DIR = ROOT_DIR / "predictions"

def check_data_files():
    """检查数据文件是否存在"""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    if not train_path.exists():
        print("❌ 训练数据文件不存在!")
        print(f"   请将 train.csv 放在 {DATA_DIR} 目录下")
        print("   或运行: python download_data.py")
        return False
    
    if not test_path.exists():
        print("⚠️  测试数据文件不存在")
        print(f"   请将 test.csv 放在 {DATA_DIR} 目录下")
        print("   或运行: python download_data.py")
        return False
    
    print("✓ 数据文件检查通过")
    return True

def quick_train():
    """快速训练模型（使用采样数据）"""
    print("\n" + "="*60)
    print("快速训练模型 (使用前10000条数据)")
    print("="*60)
    
    from src.train import main as train_main
    import sys
    
    # 设置参数
    sys.argv = [
        'train.py',
        '--train', str(DATA_DIR / "train.csv"),
        '--test', str(DATA_DIR / "test.csv"),
        '--model-type', 'xgb',
        '--cv-folds', '3',
        '--sample-size', '10000',
        '--model-path', str(MODEL_DIR / "nyc_taxi_quick.joblib"),
        '--submission', str(PRED_DIR / "submission_quick.csv"),
        '--random-state', '42'
    ]
    
    train_main()

def main():
    print("NYC Taxi Trip Duration - 快速启动")
    print("="*60)
    
    # 检查数据
    if not check_data_files():
        sys.exit(1)
    
    # 创建必要目录
    MODEL_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    
    # 快速训练
    try:
        quick_train()
        print("\n" + "="*60)
        print("✓ 快速训练完成!")
        print(f"  模型保存在: {MODEL_DIR / 'nyc_taxi_quick.joblib'}")
        print(f"  提交文件保存在: {PRED_DIR / 'submission_quick.csv'}")
        print("\n要进行完整训练，运行:")
        print("  python src/train.py --train data/train.csv --test data/test.csv")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



