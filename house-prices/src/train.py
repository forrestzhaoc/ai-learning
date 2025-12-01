#!/usr/bin/env python3
"""
House Prices Kaggle 解决方案 - 主训练脚本

用法:
    python src/train.py
    python src/train.py --model-type xgb
    python src/train.py --model-type rf --cv-folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from data_processing import prepare_data


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
PREDICTIONS_DIR = ROOT_DIR / "predictions"


def rmse_scorer(y_true, y_pred):
    """RMSE 评分器（Kaggle 使用 RMSE）"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_model(model_type: str = 'xgb', random_state: int = 42) -> object:
    """构建模型"""
    if model_type == 'xgb':
        return xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
    elif model_type == 'rf':
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == 'gbr':
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        )
    elif model_type == 'stacking':
        # 简单的 stacking 模型
        base_models = [
            ('xgb', xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                random_state=random_state,
                n_jobs=-1,
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=random_state,
                n_jobs=-1,
            )),
            ('gbr', GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_state,
            )),
        ]
        meta_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        )
        return VotingRegressor(estimators=base_models, weights=[2, 1, 1])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train House Prices model')
    parser.add_argument('--train', type=str, default=str(DATA_DIR / 'train.csv'),
                       help='Path to training CSV')
    parser.add_argument('--test', type=str, default=str(DATA_DIR / 'test.csv'),
                       help='Path to test CSV')
    parser.add_argument('--model-type', type=str, default='xgb',
                       choices=['xgb', 'rf', 'gbr', 'stacking'],
                       help='Model type to use')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state')
    parser.add_argument('--model-path', type=str,
                       default=str(MODELS_DIR / 'house_prices_model.joblib'),
                       help='Path to save model')
    parser.add_argument('--submission', type=str,
                       default=str(SUBMISSIONS_DIR / 'house_prices_submission.csv'),
                       help='Path to save submission')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("House Prices - Advanced Regression Techniques")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"交叉验证折数: {args.cv_folds}")
    print()
    
    # 读取数据
    print("读取数据...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    print()
    
    # 数据预处理
    print("数据预处理和特征工程...")
    X_train, X_test, y_train = prepare_data(train_df, test_df, target_col='SalePrice')
    print(f"特征数量: {X_train.shape[1]}")
    print(f"训练样本数: {X_train.shape[0]}")
    print()
    
    # 构建模型
    print(f"构建 {args.model_type} 模型...")
    model = build_model(args.model_type, args.random_state)
    
    # 交叉验证
    print(f"执行 {args.cv_folds} 折交叉验证...")
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    rmse_scorer_obj = make_scorer(rmse_scorer, greater_is_better=False)
    
    # 对于 XGBoost，需要特殊处理 early stopping
    if args.model_type == 'xgb':
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # 为每个 fold 创建新模型
            fold_model = build_model(args.model_type, args.random_state)
            fold_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            y_pred = fold_model.predict(X_val_fold)
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            cv_scores.append(fold_rmse)
        
        cv_scores = np.array(cv_scores)
    else:
        cv_scores = -cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring=rmse_scorer_obj,
            n_jobs=-1
        )
    
    print(f"CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    
    # 训练最终模型
    print("训练最终模型...")
    if args.model_type == 'xgb':
        # 使用部分数据作为验证集用于 early stopping
        split_idx = int(len(X_train) * 0.8)
        X_train_fit = X_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_train_fit = y_train.iloc[:split_idx]
        y_val = y_train.iloc[split_idx:]
        
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
    else:
        model.fit(X_train, y_train)
    
    # 训练集评估
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print()
    
    # 保存模型
    print(f"保存模型到: {args.model_path}")
    joblib.dump(model, args.model_path)
    print()
    
    # 预测测试集
    print("预测测试集...")
    test_pred = model.predict(X_test)
    
    # 确保预测值为正（房价不能为负）
    test_pred = np.maximum(test_pred, 0)
    
    # 创建提交文件
    test_id = test_df['Id'] if 'Id' in test_df.columns else pd.Series(range(1, len(test_df) + 1))
    submission = pd.DataFrame({
        'Id': test_id,
        'SalePrice': test_pred
    })
    
    # 保存提交文件
    print(f"保存提交文件到: {args.submission}")
    submission.to_csv(args.submission, index=False)
    print()
    
    # 保存预测结果到 predictions 目录
    predictions_path = PREDICTIONS_DIR / f"house_prices_predictions_{args.model_type}.csv"
    submission.to_csv(predictions_path, index=False)
    print(f"预测结果已保存到: {predictions_path}")
    print()
    
    print("=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

