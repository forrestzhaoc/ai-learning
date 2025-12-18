"""
大规模糖尿病数据集训练脚本
训练70万样本的数据集
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_processing_large import load_and_process_data


def train_models(X_train, y_train, X_val, y_val, use_sample=True):
    """
    训练多个模型
    
    Parameters:
    -----------
    use_sample : bool
        是否使用采样数据加速训练
    """
    print("=" * 70)
    print("开始训练模型")
    print("=" * 70)
    
    # 如果数据太大，使用采样
    if use_sample and len(X_train) > 100000:
        print(f"\n原始训练集大小: {len(X_train)}")
        print("使用采样数据加速训练...")
        sample_size = 100000
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[sample_idx]
        y_train_sample = y_train[sample_idx]
        print(f"采样后训练集大小: {len(X_train_sample)}")
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    models = {}
    scores = {}
    
    # 1. Logistic Regression
    print("\n训练 Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_sample, y_train_sample)
    
    y_pred = lr.predict(X_val)
    y_pred_proba = lr.predict_proba(X_val)[:, 1]
    
    scores['LogisticRegression'] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_pred_proba),
        'f1': f1_score(y_val, y_pred)
    }
    models['LogisticRegression'] = lr
    
    print(f"  Accuracy: {scores['LogisticRegression']['accuracy']:.4f}")
    print(f"  AUC: {scores['LogisticRegression']['auc']:.4f}")
    print(f"  F1-Score: {scores['LogisticRegression']['f1']:.4f}")
    
    # 2. Random Forest
    print("\n训练 Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_sample, y_train_sample)
    
    y_pred = rf.predict(X_val)
    y_pred_proba = rf.predict_proba(X_val)[:, 1]
    
    scores['RandomForest'] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_pred_proba),
        'f1': f1_score(y_val, y_pred)
    }
    models['RandomForest'] = rf
    
    print(f"  Accuracy: {scores['RandomForest']['accuracy']:.4f}")
    print(f"  AUC: {scores['RandomForest']['auc']:.4f}")
    print(f"  F1-Score: {scores['RandomForest']['f1']:.4f}")
    
    # 3. XGBoost
    print("\n训练 XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb.fit(X_train_sample, y_train_sample)
    
    y_pred = xgb.predict(X_val)
    y_pred_proba = xgb.predict_proba(X_val)[:, 1]
    
    scores['XGBoost'] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_pred_proba),
        'f1': f1_score(y_val, y_pred)
    }
    models['XGBoost'] = xgb
    
    print(f"  Accuracy: {scores['XGBoost']['accuracy']:.4f}")
    print(f"  AUC: {scores['XGBoost']['auc']:.4f}")
    print(f"  F1-Score: {scores['XGBoost']['f1']:.4f}")
    
    # 4. LightGBM
    print("\n训练 LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train_sample, y_train_sample)
    
    y_pred = lgbm.predict(X_val)
    y_pred_proba = lgbm.predict_proba(X_val)[:, 1]
    
    scores['LightGBM'] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_pred_proba),
        'f1': f1_score(y_val, y_pred)
    }
    models['LightGBM'] = lgbm
    
    print(f"  Accuracy: {scores['LightGBM']['accuracy']:.4f}")
    print(f"  AUC: {scores['LightGBM']['auc']:.4f}")
    print(f"  F1-Score: {scores['LightGBM']['f1']:.4f}")
    
    return models, scores


def create_ensemble(models, X_val, y_val):
    """创建集成模型"""
    print("\n" + "=" * 70)
    print("创建集成模型")
    print("=" * 70)
    
    predictions = []
    weights = []
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        predictions.append(y_pred_proba)
        auc = roc_auc_score(y_val, y_pred_proba)
        weights.append(auc)
    
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # 加权平均
    ensemble_proba = np.zeros(len(y_val))
    for pred, weight in zip(predictions, weights):
        ensemble_proba += pred * weight
    
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    ensemble_scores = {
        'accuracy': accuracy_score(y_val, ensemble_pred),
        'auc': roc_auc_score(y_val, ensemble_proba),
        'f1': f1_score(y_val, ensemble_pred)
    }
    
    print(f"\n集成模型性能:")
    print(f"  Accuracy: {ensemble_scores['accuracy']:.4f}")
    print(f"  AUC: {ensemble_scores['auc']:.4f}")
    print(f"  F1-Score: {ensemble_scores['f1']:.4f}")
    
    print(f"\n各模型权重:")
    for name, weight in zip(models.keys(), weights):
        print(f"  {name}: {weight:.4f}")
    
    return weights, ensemble_scores


def save_models(models, processor, weights, scores):
    """保存模型"""
    print("\n" + "=" * 70)
    print("保存模型")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # 保存各个模型
    for name, model in models.items():
        path = f"models/large_{name.lower()}.joblib"
        joblib.dump(model, path)
        print(f"已保存: {path}")
    
    # 保存处理器
    processor_path = 'models/large_processor.joblib'
    joblib.dump(processor, processor_path)
    print(f"已保存: {processor_path}")
    
    # 保存集成权重
    ensemble_info = {
        'weights': weights,
        'model_names': list(models.keys())
    }
    ensemble_path = 'models/large_ensemble_weights.joblib'
    joblib.dump(ensemble_info, ensemble_path)
    print(f"已保存: {ensemble_path}")
    
    # 保存性能分数
    scores_df = pd.DataFrame(scores).T
    scores_path = 'models/large_model_scores.csv'
    scores_df.to_csv(scores_path)
    print(f"已保存: {scores_path}")


def print_summary(scores):
    """打印训练总结"""
    print("\n" + "=" * 70)
    print("训练总结")
    print("=" * 70)
    
    scores_df = pd.DataFrame(scores).T
    scores_df = scores_df.sort_values('auc', ascending=False)
    print("\n模型性能排名（按AUC）:")
    print(scores_df.to_string())


def main():
    """主函数"""
    print("=" * 70)
    print("大规模糖尿病数据集训练")
    print("=" * 70)
    
    # 加载和处理数据
    print("\n加载数据...")
    X_train, y_train, X_test, feature_names, processor, test_ids = \
        load_and_process_data()
    
    # 分割训练集和验证集
    print("\n分割训练集和验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    print(f"训练集: {X_train_split.shape}")
    print(f"验证集: {X_val.shape}")
    
    # 训练模型
    models, scores = train_models(X_train_split, y_train_split, X_val, y_val)
    
    # 创建集成模型
    weights, ensemble_scores = create_ensemble(models, X_val, y_val)
    scores['Ensemble'] = ensemble_scores
    
    # 打印总结
    print_summary(scores)
    
    # 保存模型
    save_models(models, processor, weights, scores)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print("\n接下来运行: python3 generate_large_submission.py")


if __name__ == '__main__':
    main()








