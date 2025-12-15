"""
快速开始脚本 - Transformer版本
"""

import os
import sys
from datetime import datetime
import torch

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_data_exists():
    """Check if data files exist"""
    data_dir = 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return False
    return True

def run_transformer_pipeline(n_folds=5, epochs=50, batch_size=512):
    """
    运行Transformer模型完整流程
    """
    print_header("ROAD ACCIDENT RISK PREDICTION - TRANSFORMER MODEL")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查数据
    print_header("Step 1: Data Check")
    if not check_data_exists():
        print("❌ 数据文件未找到！")
        print("请先下载数据:")
        print("  python download_data.py")
        sys.exit(1)
    print("✓ 数据文件存在")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 训练模型
    print_header("Step 2: Model Training")
    from train_transformer import train_with_transformer
    
    try:
        model_path, oof_rmse, fold_models = train_with_transformer(
            n_folds=n_folds,
            batch_size=batch_size,
            num_epochs=epochs,
            learning_rate=1e-4,
            device=device
        )
        print("\n✓ 模型训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 生成提交文件
    print_header("Step 3: Generate Submission")
    from train_transformer import generate_submission_transformer
    
    try:
        submission_path = generate_submission_transformer(
            model_path=model_path,
            batch_size=batch_size,
            device=device
        )
        print("\n✓ 提交文件生成完成")
    except Exception as e:
        print(f"❌ 生成提交文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 总结
    print_header("PIPELINE COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n生成的文件:")
    print(f"  - {model_path} (最终模型)")
    print(f"  - {submission_path} (提交文件)")
    print(f"\n模型性能:")
    print(f"  OOF RMSE: {oof_rmse:.6f}")
    print(f"\n提交到Kaggle:")
    print(f"  kaggle competitions submit -c playground-series-s5e10 \\")
    print(f"    -f {submission_path} -m \"Transformer model (RMSE: {oof_rmse:.6f})\"")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start Transformer pipeline')
    parser.add_argument('--folds', type=int, default=5, help='CV folds')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    
    args = parser.parse_args()
    
    try:
        run_transformer_pipeline(
            n_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n\n流程被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n流程失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
