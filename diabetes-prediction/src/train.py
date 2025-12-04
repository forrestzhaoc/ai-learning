"""
模型训练脚本
使用多种算法进行训练和集成
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from data_processing import load_and_process_data, split_train_val


class DiabetesModelTrainer:
    """糖尿病预测模型训练器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
    
    def get_models(self):
        """定义所有模型"""
        models = {
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=0.1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=self.random_state
            )
        }
        return models
    
    def train_single_model(self, model_name, model, X_train, y_train, X_val, y_val):
        """训练单个模型"""
        print(f"\n训练 {model_name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 验证集预测
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        
        print(f"  验证集 Accuracy: {accuracy:.4f}")
        print(f"  验证集 AUC: {auc:.4f}")
        print(f"  验证集 F1-Score: {f1:.4f}")
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                    scoring='roc_auc', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  交叉验证 AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # 保存模型和分数
        self.models[model_name] = model
        self.model_scores[model_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'cv_auc_mean': cv_mean,
            'cv_auc_std': cv_std
        }
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """训练所有模型"""
        print("=" * 60)
        print("开始训练所有模型")
        print("=" * 60)
        
        models = self.get_models()
        
        for model_name, model in models.items():
            self.train_single_model(model_name, model, X_train, y_train, X_val, y_val)
        
        # 找出最佳模型
        best_auc = 0
        for model_name, scores in self.model_scores.items():
            if scores['auc'] > best_auc:
                best_auc = scores['auc']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print("\n" + "=" * 60)
        print("模型训练完成")
        print("=" * 60)
        print(f"\n最佳模型: {self.best_model_name}")
        print(f"最佳AUC: {self.model_scores[self.best_model_name]['auc']:.4f}")
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """训练集成模型（加权平均）"""
        print("\n" + "=" * 60)
        print("创建集成模型")
        print("=" * 60)
        
        # 使用所有模型进行预测
        val_predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            val_predictions.append(y_pred_proba)
            # 使用AUC作为权重
            weights.append(self.model_scores[model_name]['auc'])
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加权平均
        ensemble_pred_proba = np.zeros(len(y_val))
        for pred, weight in zip(val_predictions, weights):
            ensemble_pred_proba += pred * weight
        
        ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
        
        # 计算集成模型的指标
        accuracy = accuracy_score(y_val, ensemble_pred)
        auc = roc_auc_score(y_val, ensemble_pred_proba)
        f1 = f1_score(y_val, ensemble_pred)
        
        print(f"\n集成模型验证集结果:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        print(f"\n各模型权重:")
        for model_name, weight in zip(self.models.keys(), weights):
            print(f"  {model_name}: {weight:.4f}")
        
        # 保存集成模型信息
        self.ensemble_weights = weights
        self.model_scores['Ensemble'] = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1
        }
        
        return ensemble_pred_proba
    
    def save_models(self, save_dir='../models'):
        """保存所有模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("保存模型")
        print("=" * 60)
        
        # 保存各个模型
        for model_name, model in self.models.items():
            model_path = f"{save_dir}/diabetes_{model_name.lower()}.joblib"
            joblib.dump(model, model_path)
            print(f"已保存: {model_path}")
        
        # 保存集成权重
        ensemble_info = {
            'weights': self.ensemble_weights,
            'model_names': list(self.models.keys())
        }
        ensemble_path = f"{save_dir}/diabetes_ensemble_weights.joblib"
        joblib.dump(ensemble_info, ensemble_path)
        print(f"已保存: {ensemble_path}")
        
        # 保存最佳模型
        best_model_path = f"{save_dir}/diabetes_best_model.joblib"
        joblib.dump(self.best_model, best_model_path)
        print(f"已保存最佳模型: {best_model_path}")
        
        # 保存模型分数
        scores_df = pd.DataFrame(self.model_scores).T
        scores_path = f"{save_dir}/model_scores.csv"
        scores_df.to_csv(scores_path)
        print(f"已保存模型分数: {scores_path}")
    
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 60)
        print("训练总结")
        print("=" * 60)
        
        scores_df = pd.DataFrame(self.model_scores).T
        scores_df = scores_df.sort_values('auc', ascending=False)
        print("\n模型性能排名（按AUC）:")
        print(scores_df.to_string())


def main():
    """主函数"""
    print("=" * 60)
    print("糖尿病预测模型训练")
    print("=" * 60)
    
    # 加载和处理数据
    print("\n加载数据...")
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_path = os.path.join(project_dir, 'data', 'train.csv')
    test_path = os.path.join(project_dir, 'data', 'test.csv')
    
    X_train, y_train, X_test, feature_names, processor = load_and_process_data(
        train_path=train_path,
        test_path=test_path,
        handle_zeros='median',
        scaler_type='standard'
    )
    
    # 分割训练集和验证集
    print("\n分割训练集和验证集...")
    X_train_split, X_val, y_train_split, y_val = split_train_val(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {X_train_split.shape}")
    print(f"验证集: {X_val.shape}")
    
    # 创建训练器
    trainer = DiabetesModelTrainer(random_state=42)
    
    # 训练所有模型
    trainer.train_all_models(X_train_split, y_train_split, X_val, y_val)
    
    # 训练集成模型
    trainer.train_ensemble(X_train_split, y_train_split, X_val, y_val)
    
    # 打印总结
    trainer.print_summary()
    
    # 保存模型
    models_dir = os.path.join(project_dir, 'models')
    trainer.save_models(save_dir=models_dir)
    
    # 保存数据处理器
    processor_path = os.path.join(models_dir, 'diabetes_processor.joblib')
    joblib.dump(processor, processor_path)
    print(f"\n已保存数据处理器: {processor_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

