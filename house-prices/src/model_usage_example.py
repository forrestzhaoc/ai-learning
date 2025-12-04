#!/usr/bin/env python3
"""
XGBoost 模型使用示例
演示如何加载模型、预测、查看特征重要性等
"""
from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

from data_processing import prepare_data

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


def load_model_example():
    """示例1：加载保存的模型"""
    print("=" * 60)
    print("示例1：加载保存的模型")
    print("=" * 60)
    
    # 加载模型
    model_path = MODELS_DIR / 'house_prices_model.joblib'
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✓ 模型已加载：{model_path}")
        print(f"  模型类型：{type(model)}")
        print(f"  树的数量：{model.n_estimators}")
        print(f"  最大深度：{model.max_depth}")
        print(f"  学习率：{model.learning_rate}")
    else:
        print(f"✗ 模型文件不存在：{model_path}")
        print("  请先运行 python src/train.py 训练模型")
    
    print()


def predict_example():
    """示例2：使用模型预测新数据"""
    print("=" * 60)
    print("示例2：使用模型预测新数据")
    print("=" * 60)
    
    # 加载模型
    model_path = MODELS_DIR / 'house_prices_model.joblib'
    if not model_path.exists():
        print(f"✗ 模型文件不存在：{model_path}")
        return
    
    model = joblib.load(model_path)
    
    # 加载测试数据
    test_file = DATA_DIR / 'test.csv'
    if not test_file.exists():
        print(f"✗ 测试数据文件不存在：{test_file}")
        return
    
    test_df = pd.read_csv(test_file)
    print(f"✓ 测试数据已加载：{test_df.shape[0]} 个样本")
    
    # 准备数据（需要和训练时相同的预处理）
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    _, X_test, _ = prepare_data(train_df, test_df, target_col='SalePrice')
    
    # 预测
    print("正在预测...")
    predictions = model.predict(X_test)
    
    # 后处理：确保非负
    predictions = np.maximum(predictions, 0)
    
    print(f"✓ 预测完成")
    print(f"  预测数量：{len(predictions)}")
    print(f"  价格范围：${predictions.min():,.0f} - ${predictions.max():,.0f}")
    print(f"  平均价格：${predictions.mean():,.0f}")
    print(f"  中位数：${np.median(predictions):,.0f}")
    
    # 显示前5个预测
    print("\n前5个预测：")
    for i in range(min(5, len(predictions))):
        print(f"  样本 {i+1}: ${predictions[i]:,.0f}")
    
    print()


def feature_importance_example():
    """示例3：查看特征重要性"""
    print("=" * 60)
    print("示例3：查看特征重要性")
    print("=" * 60)
    
    # 加载模型
    model_path = MODELS_DIR / 'house_prices_model.joblib'
    if not model_path.exists():
        print(f"✗ 模型文件不存在：{model_path}")
        return
    
    model = joblib.load(model_path)
    
    # 加载训练数据（用于获取特征名称）
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    X_train, _, _ = prepare_data(train_df, test_df, target_col='SalePrice')
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"✓ 特征重要性已计算")
    print(f"  总特征数：{len(feature_importance)}")
    print()
    
    # 显示前10个最重要的特征
    print("前10个最重要的特征：")
    print("-" * 60)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30s} {row['importance']:8.6f}")
    
    print()
    
    # 显示重要性分布
    print("重要性分布：")
    print(f"  最大值：{feature_importance['importance'].max():.6f}")
    print(f"  最小值：{feature_importance['importance'].min():.6f}")
    print(f"  平均值：{feature_importance['importance'].mean():.6f}")
    print(f"  中位数：{feature_importance['importance'].median():.6f}")
    
    print()


def model_info_example():
    """示例4：查看模型详细信息"""
    print("=" * 60)
    print("示例4：查看模型详细信息")
    print("=" * 60)
    
    # 加载模型
    model_path = MODELS_DIR / 'house_prices_model.joblib'
    if not model_path.exists():
        print(f"✗ 模型文件不存在：{model_path}")
        return
    
    model = joblib.load(model_path)
    
    print("模型参数：")
    print("-" * 60)
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  subsample: {model.subsample}")
    print(f"  colsample_bytree: {model.colsample_bytree}")
    print(f"  random_state: {model.random_state}")
    print(f"  n_jobs: {model.n_jobs}")
    print()
    
    # 获取实际训练的树数量（如果有early stopping）
    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        print(f"  实际训练的树数：{model.best_iteration + 1}")
        print(f"  （Early Stopping 在第 {model.best_iteration + 1} 轮停止）")
    else:
        print(f"  实际训练的树数：{model.n_estimators}")
    
    print()


def single_prediction_example():
    """示例5：预测单个样本"""
    print("=" * 60)
    print("示例5：预测单个样本")
    print("=" * 60)
    
    # 加载模型
    model_path = MODELS_DIR / 'house_prices_model.joblib'
    if not model_path.exists():
        print(f"✗ 模型文件不存在：{model_path}")
        return
    
    model = joblib.load(model_path)
    
    # 加载训练数据（用于获取特征名称和预处理）
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    X_train, X_test, y_train = prepare_data(train_df, test_df, target_col='SalePrice')
    
    # 选择一个样本（第一个测试样本）
    sample = X_test.iloc[0:1]
    
    # 预测
    prediction = model.predict(sample)[0]
    prediction = max(prediction, 0)  # 确保非负
    
    print(f"✓ 预测完成")
    print(f"  预测价格：${prediction:,.0f}")
    print()
    
    # 显示样本的关键特征
    print("样本的关键特征：")
    print("-" * 60)
    key_features = ['TotalSF', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBathrooms']
    for feat in key_features:
        if feat in sample.columns:
            print(f"  {feat}: {sample[feat].values[0]}")
    
    print()


def main():
    """运行所有示例"""
    print()
    print("=" * 60)
    print("XGBoost 模型使用示例")
    print("=" * 60)
    print()
    
    # 运行示例
    load_model_example()
    predict_example()
    feature_importance_example()
    model_info_example()
    single_prediction_example()
    
    print("=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()


