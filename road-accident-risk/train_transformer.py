"""
使用Transformer模型训练道路事故风险预测
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import os
from datetime import datetime

from src.data_processing import DataProcessor
from src.transformer_model import train_transformer, predict_transformer

def train_with_transformer(
    data_dir='data',
    n_folds=5,
    batch_size=512,
    num_epochs=50,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='models'
):
    """
    使用Transformer模型进行K折交叉验证训练
    
    Args:
        data_dir: 数据目录
        n_folds: 交叉验证折数
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        save_dir: 模型保存目录
    """
    print("\n" + "="*70)
    print("Transformer Model Training - Road Accident Risk Prediction")
    print("="*70)
    
    # 加载数据
    processor = DataProcessor(data_dir=data_dir)
    train_df, test_df = processor.load_data()
    
    # 准备特征
    X_train_full, X_test, y_train_full, test_ids = processor.prepare_features(
        train_df, test_df, target_col='accident_risk'
    )
    
    print(f"\n数据准备完成:")
    print(f"  训练集: {X_train_full.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  特征数: {X_train_full.shape[1]}")
    print(f"  设备: {device}")
    
    # K折交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X_train_full))
    fold_models = []
    fold_scores = []
    
    print(f"\n{'='*70}")
    print(f"开始 {n_folds} 折交叉验证训练")
    print(f"{'='*70}\n")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        print(f"训练集: {X_train.shape[0]:,} 样本")
        print(f"验证集: {X_val.shape[0]:,} 样本")
        
        # 模型参数（优化内存使用）
        model_params = {
            'input_dim': X_train.shape[1],
            'embedding_dim': 64,  # 减小embedding维度
            'num_heads': 4,  # 减少注意力头数
            'num_layers': 4,  # 减少层数
            'hidden_dim': 256,  # 减小隐藏层
            'dropout': 0.1,
            'dim_feedforward': 512  # 减小前馈网络
        }
        
        # 训练模型
        model_path = os.path.join(save_dir, f'transformer_fold_{fold+1}.pth')
        
        train_transformer(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            model_params=model_params,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            save_path=model_path
        )
        
        # 验证集预测
        val_pred = predict_transformer(
            X_val,
            model_path=model_path,
            batch_size=batch_size,
            device=device
        )
        
        oof_predictions[val_idx] = val_pred
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_mae = mean_absolute_error(y_val, val_pred)
        
        fold_scores.append({
            'fold': fold + 1,
            'rmse': fold_rmse,
            'mae': fold_mae
        })
        
        print(f"\nFold {fold + 1} 验证结果:")
        print(f"  RMSE: {fold_rmse:.6f}")
        print(f"  MAE: {fold_mae:.6f}")
        
        fold_models.append(model_path)
    
    # 计算OOF分数
    oof_rmse = np.sqrt(mean_squared_error(y_train_full, oof_predictions))
    oof_mae = mean_absolute_error(y_train_full, oof_predictions)
    
    print(f"\n{'='*70}")
    print("交叉验证结果汇总")
    print(f"{'='*70}")
    
    for score in fold_scores:
        print(f"Fold {score['fold']}: RMSE = {score['rmse']:.6f}, MAE = {score['mae']:.6f}")
    
    print(f"\n总体 OOF 分数:")
    print(f"  RMSE: {oof_rmse:.6f}")
    print(f"  MAE: {oof_mae:.6f}")
    
    # 使用全部数据重新训练最终模型
    print(f"\n{'='*70}")
    print("使用全部数据训练最终模型")
    print(f"{'='*70}\n")
    
    final_model_params = {
        'input_dim': X_train_full.shape[1],
        'embedding_dim': 64,
        'num_heads': 4,
        'num_layers': 4,
        'hidden_dim': 256,
        'dropout': 0.1,
        'dim_feedforward': 512
    }
    
    final_model_path = os.path.join(save_dir, 'transformer_final.pth')
    
    train_transformer(
        X_train_full, y_train_full,
        X_val=None, y_val=None,
        model_params=final_model_params,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=final_model_path
    )
    
    print(f"\n{'='*70}")
    print("训练完成！")
    print(f"{'='*70}")
    print(f"最终模型保存至: {final_model_path}")
    print(f"交叉验证 RMSE: {oof_rmse:.6f}")
    print(f"交叉验证 MAE: {oof_mae:.6f}")
    
    return final_model_path, oof_rmse, fold_models


def generate_submission_transformer(
    model_path='models/transformer_final.pth',
    data_dir='data',
    output_dir='predictions',
    batch_size=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    使用Transformer模型生成提交文件
    """
    print("\n" + "="*70)
    print("生成提交文件 - Transformer模型")
    print("="*70)
    
    # 加载数据
    processor = DataProcessor(data_dir=data_dir)
    train_df, test_df = processor.load_data()
    
    # 准备特征
    X_train, X_test, y_train, test_ids = processor.prepare_features(
        train_df, test_df, target_col='accident_risk'
    )
    
    print(f"\n测试集: {X_test.shape[0]:,} 样本")
    
    # 预测
    print("\n开始预测...")
    predictions = predict_transformer(
        X_test,
        model_path=model_path,
        batch_size=batch_size,
        device=device
    )
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_ids,
        'accident_risk': predictions
    })
    
    # 确保预测值在合理范围内
    submission_df['accident_risk'] = submission_df['accident_risk'].clip(lower=0, upper=1)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'submission_transformer_{timestamp}.csv')
    
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✓ 提交文件已生成: {output_path}")
    print(f"  预测数量: {len(submission_df):,}")
    print(f"\n预测统计:")
    print(submission_df['accident_risk'].describe())
    
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Transformer model for Road Accident Risk Prediction')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--generate-submission', action='store_true', help='Generate submission after training')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练模型
    model_path, oof_rmse, fold_models = train_with_transformer(
        n_folds=args.folds,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    # 生成提交文件
    if args.generate_submission:
        generate_submission_transformer(
            model_path=model_path,
            batch_size=args.batch_size,
            device=device
        )
