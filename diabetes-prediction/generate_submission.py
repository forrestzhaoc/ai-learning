"""
生成Kaggle提交文件
使用训练好的模型对测试集进行预测
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append('src')

from data_processing import DiabetesDataProcessor


def load_test_data(test_path='data/test.csv'):
    """加载测试数据"""
    return pd.read_csv(test_path)


def generate_submission_single_model(model_name='best_model', 
                                     model_path=None,
                                     output_name=None):
    """使用单个模型生成提交文件"""
    print(f"=" * 60)
    print(f"使用 {model_name} 生成提交文件")
    print(f"=" * 60)
    
    # 加载模型和处理器
    if model_path is None:
        model_path = f'models/diabetes_{model_name}.joblib'
    
    processor_path = 'models/diabetes_processor.joblib'
    
    print(f"\n加载模型: {model_path}")
    model = joblib.load(model_path)
    
    print(f"加载数据处理器: {processor_path}")
    processor = joblib.load(processor_path)
    
    # 加载测试数据
    print(f"\n加载测试数据...")
    test_df = pd.read_csv('data/test.csv')
    print(f"测试集形状: {test_df.shape}")
    
    # 保存ID列（如果存在）
    if 'Id' in test_df.columns:
        test_ids = test_df['Id'].values
    else:
        test_ids = np.arange(len(test_df))
    
    # 处理测试数据
    print(f"\n处理测试数据...")
    X_test, feature_names = processor.transform(test_df, is_test=True)
    print(f"处理后的测试集形状: {X_test.shape}")
    
    # 预测
    print(f"\n进行预测...")
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)[:, 1]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'Outcome': predictions
    })
    
    # 保存提交文件
    os.makedirs('submissions', exist_ok=True)
    
    if output_name is None:
        output_name = f'diabetes_submission_{model_name}.csv'
    
    submission_path = f'submissions/{output_name}'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n提交文件已保存: {submission_path}")
    print(f"预测统计:")
    print(f"  预测为0（无糖尿病）: {(predictions == 0).sum()} ({(predictions == 0).sum() / len(predictions) * 100:.2f}%)")
    print(f"  预测为1（有糖尿病）: {(predictions == 1).sum()} ({(predictions == 1).sum() / len(predictions) * 100:.2f}%)")
    print(f"  平均预测概率: {predictions_proba.mean():.4f}")
    
    return submission


def generate_submission_ensemble():
    """使用集成模型生成提交文件"""
    print(f"=" * 60)
    print(f"使用集成模型生成提交文件")
    print(f"=" * 60)
    
    # 加载数据处理器
    processor_path = 'models/diabetes_processor.joblib'
    print(f"加载数据处理器: {processor_path}")
    processor = joblib.load(processor_path)
    
    # 加载集成权重
    ensemble_path = 'models/diabetes_ensemble_weights.joblib'
    print(f"加载集成权重: {ensemble_path}")
    ensemble_info = joblib.load(ensemble_path)
    weights = ensemble_info['weights']
    model_names = ensemble_info['model_names']
    
    # 加载所有模型
    print(f"\n加载模型:")
    models = {}
    for model_name in model_names:
        model_path = f'models/diabetes_{model_name.lower()}.joblib'
        models[model_name] = joblib.load(model_path)
        print(f"  - {model_name}")
    
    # 加载测试数据
    print(f"\n加载测试数据...")
    test_df = pd.read_csv('data/test.csv')
    print(f"测试集形状: {test_df.shape}")
    
    # 保存ID列
    if 'Id' in test_df.columns:
        test_ids = test_df['Id'].values
    else:
        test_ids = np.arange(len(test_df))
    
    # 处理测试数据
    print(f"\n处理测试数据...")
    X_test, feature_names = processor.transform(test_df, is_test=True)
    print(f"处理后的测试集形状: {X_test.shape}")
    
    # 集成预测
    print(f"\n进行集成预测...")
    ensemble_proba = np.zeros(len(X_test))
    
    for model_name, model, weight in zip(model_names, models.values(), weights):
        pred_proba = model.predict_proba(X_test)[:, 1]
        ensemble_proba += pred_proba * weight
        print(f"  {model_name}: 权重 = {weight:.4f}")
    
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'Outcome': ensemble_pred
    })
    
    # 保存提交文件
    os.makedirs('submissions', exist_ok=True)
    submission_path = 'submissions/diabetes_submission_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n提交文件已保存: {submission_path}")
    print(f"预测统计:")
    print(f"  预测为0（无糖尿病）: {(ensemble_pred == 0).sum()} ({(ensemble_pred == 0).sum() / len(ensemble_pred) * 100:.2f}%)")
    print(f"  预测为1（有糖尿病）: {(ensemble_pred == 1).sum()} ({(ensemble_pred == 1).sum() / len(ensemble_pred) * 100:.2f}%)")
    print(f"  平均预测概率: {ensemble_proba.mean():.4f}")
    
    return submission


def main():
    """主函数"""
    print("=" * 60)
    print("生成Kaggle提交文件")
    print("=" * 60)
    
    # 检查必要文件
    required_files = [
        'data/test.csv',
        'models/diabetes_processor.joblib',
        'models/diabetes_best_model.joblib'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\n错误：缺少必要文件:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请先运行 'python src/train.py' 训练模型。")
        return
    
    print("\n选择提交方式:")
    print("1. 使用最佳单模型")
    print("2. 使用集成模型（推荐）")
    print("3. 生成所有提交文件")
    
    choice = input("\n请输入选择 (1/2/3，默认为2): ").strip()
    
    if choice == '1':
        # 使用最佳模型
        generate_submission_single_model(
            model_name='best_model',
            model_path='models/diabetes_best_model.joblib',
            output_name='diabetes_submission_best.csv'
        )
    elif choice == '3':
        # 生成所有提交文件
        print("\n生成所有提交文件...\n")
        
        # 最佳模型
        generate_submission_single_model(
            model_name='best_model',
            model_path='models/diabetes_best_model.joblib',
            output_name='diabetes_submission_best.csv'
        )
        
        print("\n" + "=" * 60 + "\n")
        
        # 集成模型
        generate_submission_ensemble()
        
        print("\n" + "=" * 60 + "\n")
        
        # 单个模型
        model_files = [
            ('logisticregression', 'LogisticRegression'),
            ('randomforest', 'RandomForest'),
            ('xgboost', 'XGBoost'),
            ('lightgbm', 'LightGBM')
        ]
        
        for model_file, model_display in model_files:
            model_path = f'models/diabetes_{model_file}.joblib'
            if os.path.exists(model_path):
                generate_submission_single_model(
                    model_name=model_file,
                    model_path=model_path,
                    output_name=f'diabetes_submission_{model_file}.csv'
                )
                print("\n" + "=" * 60 + "\n")
    
    else:
        # 默认使用集成模型
        generate_submission_ensemble()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n提交文件位于 'submissions/' 目录")
    print("请登录Kaggle并上传相应的CSV文件。")


if __name__ == '__main__':
    main()

