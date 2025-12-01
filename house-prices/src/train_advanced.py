#!/usr/bin/env python3
"""
House Prices Kaggle 解决方案 - 高级版本

改进点：
1. 对目标变量进行 log 转换，减少偏态分布的影响
2. 更丰富的特征工程
3. 使用 Stacking 集成学习
4. 更细致的数据预处理

用法:
    python src/train_advanced.py
    python src/train_advanced.py --model-type stacking
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from data_processing_advanced import prepare_data_advanced


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
PREDICTIONS_DIR = ROOT_DIR / "predictions"


def rmse_scorer(y_true, y_pred):
    """RMSE 评分器"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def rmsle_scorer(y_true, y_pred):
    """RMSLE 评分器（用于 log 转换的目标变量）"""
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def build_base_models(random_state: int = 42) -> dict:
    """构建基础模型"""
    models = {
        'ridge': Ridge(alpha=10.0, random_state=random_state),
        'lasso': Lasso(alpha=0.0005, random_state=random_state, max_iter=10000),
        'elasticnet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        'xgb': xgb.XGBRegressor(
            n_estimators=2000,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        ),
        'rf': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
        ),
        'gbr': GradientBoostingRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            random_state=random_state,
        ),
    }
    return models


def train_stacking_model(X_train, y_train, cv_folds: int = 5, random_state: int = 42):
    """训练 Stacking 模型"""
    print("构建 Stacking 模型...")
    
    # 获取基础模型
    base_models = build_base_models(random_state)
    
    # 第一层模型预测
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 存储每个基础模型的预测结果
    base_train_preds = {}
    
    for name, model in base_models.items():
        print(f"  训练基础模型: {name}")
        
        # 交叉验证生成训练集预测
        fold_preds = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            fold_preds[val_idx] = model.predict(X_fold_val)
        
        base_train_preds[name] = fold_preds
        
        # 在完整训练集上训练最终模型
        model.fit(X_train, y_train)
    
    # 构建第二层训练数据
    stacked_X_train = pd.DataFrame(base_train_preds)
    
    # 第二层模型（元模型）
    meta_model = Ridge(alpha=5.0, random_state=random_state)
    meta_model.fit(stacked_X_train, y_train)
    
    print("  Stacking 模型训练完成")
    
    return base_models, meta_model


def predict_stacking(base_models, meta_model, X_test):
    """使用 Stacking 模型进行预测"""
    # 获取基础模型预测
    base_preds = {}
    for name, model in base_models.items():
        base_preds[name] = model.predict(X_test)
    
    # 构建第二层输入
    stacked_X_test = pd.DataFrame(base_preds)
    
    # 元模型预测
    final_pred = meta_model.predict(stacked_X_test)
    
    return final_pred


def main():
    parser = argparse.ArgumentParser(description='Train House Prices model (Advanced)')
    parser.add_argument('--train', type=str, default=str(DATA_DIR / 'train.csv'),
                       help='Path to training CSV')
    parser.add_argument('--test', type=str, default=str(DATA_DIR / 'test.csv'),
                       help='Path to test CSV')
    parser.add_argument('--model-type', type=str, default='stacking',
                       choices=['stacking', 'xgb', 'ensemble'],
                       help='Model type to use')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state')
    parser.add_argument('--use-log', action='store_true', default=True,
                       help='Use log transformation for target variable')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("House Prices - Advanced Regression Techniques (Advanced Version)")
    print("=" * 70)
    print(f"模型类型: {args.model_type}")
    print(f"交叉验证折数: {args.cv_folds}")
    print(f"使用对数转换: {args.use_log}")
    print()
    
    # 读取数据
    print("读取数据...")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    test_ids = test_df['Id'].copy()
    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    print()
    
    # 数据预处理（使用高级版本）
    print("数据预处理和特征工程（高级版本）...")
    X_train, X_test, y_train = prepare_data_advanced(train_df, test_df)
    
    # 对目标变量进行 log 转换
    if args.use_log:
        y_train_log = np.log1p(y_train)
        print("已对目标变量进行 log1p 转换")
    else:
        y_train_log = y_train
    
    print(f"特征数量: {X_train.shape[1]}")
    print(f"训练样本数: {X_train.shape[0]}")
    print()
    
    # 训练模型
    if args.model_type == 'stacking':
        # Stacking 模型
        base_models, meta_model = train_stacking_model(
            X_train, y_train_log, 
            cv_folds=args.cv_folds,
            random_state=args.random_state
        )
        
        # 交叉验证评估
        print("\n执行交叉验证...")
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train_log.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train_log.iloc[val_idx]
            
            # 训练临时的 stacking 模型
            temp_base_models = build_base_models(args.random_state)
            for model in temp_base_models.values():
                model.fit(X_fold_train, y_fold_train)
            
            # 预测
            base_preds = {}
            for name, model in temp_base_models.items():
                base_preds[name] = model.predict(X_fold_val)
            
            stacked_X_val = pd.DataFrame(base_preds)
            temp_meta = Ridge(alpha=5.0, random_state=args.random_state)
            
            # 训练元模型
            base_train_preds = {}
            for name, model in temp_base_models.items():
                base_train_preds[name] = model.predict(X_fold_train)
            stacked_X_train = pd.DataFrame(base_train_preds)
            temp_meta.fit(stacked_X_train, y_fold_train)
            
            y_pred = temp_meta.predict(stacked_X_val)
            fold_rmse = rmse_scorer(y_fold_val, y_pred)
            cv_scores.append(fold_rmse)
            print(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
        
        cv_scores = np.array(cv_scores)
        print(f"\nCV RMSE (log scale): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 训练集评估
        train_pred_log = predict_stacking(base_models, meta_model, X_train)
        train_rmse = rmse_scorer(y_train_log, train_pred_log)
        print(f"训练集 RMSE (log scale): {train_rmse:.4f}")
        
        # 预测测试集
        print("\n预测测试集...")
        test_pred_log = predict_stacking(base_models, meta_model, X_test)
        
        # 保存模型
        model_path = MODELS_DIR / 'house_prices_stacking_advanced.joblib'
        joblib.dump({'base_models': base_models, 'meta_model': meta_model}, model_path)
        print(f"模型已保存: {model_path}")
        
    elif args.model_type == 'xgb':
        # 单一 XGBoost 模型
        print("训练 XGBoost 模型...")
        model = xgb.XGBRegressor(
            n_estimators=3000,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=args.random_state,
            n_jobs=-1,
        )
        
        # 使用部分数据作为验证集
        split_idx = int(len(X_train) * 0.85)
        X_train_fit = X_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_train_fit = y_train_log.iloc[:split_idx]
        y_val = y_train_log.iloc[split_idx:]
        
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        # 交叉验证
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        rmse_scorer_obj = make_scorer(rmse_scorer, greater_is_better=False)
        cv_scores = -cross_val_score(model, X_train, y_train_log, cv=cv, scoring=rmse_scorer_obj, n_jobs=-1)
        print(f"\nCV RMSE (log scale): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        test_pred_log = model.predict(X_test)
        
        model_path = MODELS_DIR / 'house_prices_xgb_advanced.joblib'
        joblib.dump(model, model_path)
        
    else:  # ensemble
        # 简单平均集成
        print("训练集成模型（简单平均）...")
        models = build_base_models(args.random_state)
        
        for name, model in models.items():
            print(f"  训练模型: {name}")
            model.fit(X_train, y_train_log)
        
        # 预测
        test_preds = []
        for name, model in models.items():
            pred = model.predict(X_test)
            test_preds.append(pred)
        
        test_pred_log = np.mean(test_preds, axis=0)
        
        model_path = MODELS_DIR / 'house_prices_ensemble_advanced.joblib'
        joblib.dump(models, model_path)
    
    # 反转 log 转换
    if args.use_log:
        test_pred = np.expm1(test_pred_log)
    else:
        test_pred = test_pred_log
    
    # 确保预测值为正
    test_pred = np.maximum(test_pred, 0)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_pred
    })
    
    # 保存提交文件
    submission_path = SUBMISSIONS_DIR / f'house_prices_submission_{args.model_type}_advanced.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n提交文件已保存: {submission_path}")
    
    # 保存预测结果
    predictions_path = PREDICTIONS_DIR / f'house_prices_predictions_{args.model_type}_advanced.csv'
    submission.to_csv(predictions_path, index=False)
    print(f"预测结果已保存: {predictions_path}")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("预测统计信息:")
    print(f"  记录数: {len(submission)}")
    print(f"  价格范围: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
    print(f"  平均价格: ${submission['SalePrice'].mean():,.0f}")
    print(f"  中位数价格: ${submission['SalePrice'].median():,.0f}")
    print("=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()

