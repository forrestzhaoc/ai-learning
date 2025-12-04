"""
生成大规模数据集的提交文件
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append('src')
import warnings
warnings.filterwarnings('ignore')

from data_processing_large import LargeDiabetesDataProcessor


def generate_submission(model_name='ensemble'):
    """
    生成提交文件
    
    Parameters:
    -----------
    model_name : str
        模型名称: 'ensemble', 'lightgbm', 'xgboost', 'randomforest', 'logisticregression'
    """
    print("=" * 70)
    print(f"生成提交文件 - {model_name}")
    print("=" * 70)
    
    # 加载数据处理器
    print("\n加载数据处理器...")
    processor = joblib.load('models/large_processor.joblib')
    
    # 加载测试数据
    print("加载测试数据...")
    test_df = pd.read_csv('data/test.csv')
    print(f"测试集形状: {test_df.shape}")
    
    # 处理测试数据
    print("处理测试数据...")
    X_test, feature_names, test_ids = processor.transform(test_df, is_test=True)
    print(f"处理后形状: {X_test.shape}")
    
    # 预测
    if model_name.lower() == 'ensemble':
        print("\n使用集成模型进行预测...")
        
        # 加载集成权重
        ensemble_info = joblib.load('models/large_ensemble_weights.joblib')
        weights = ensemble_info['weights']
        model_names = ensemble_info['model_names']
        
        # 加载所有模型并预测
        ensemble_proba = np.zeros(len(X_test))
        
        for name, weight in zip(model_names, weights):
            model_path = f'models/large_{name.lower()}.joblib'
            model = joblib.load(model_path)
            proba = model.predict_proba(X_test)[:, 1]
            ensemble_proba += proba * weight
            print(f"  {name}: 权重 = {weight:.4f}")
        
        predictions = (ensemble_proba >= 0.5).astype(int)
        proba = ensemble_proba
        
    else:
        print(f"\n使用 {model_name} 进行预测...")
        model_path = f'models/large_{model_name.lower()}.joblib'
        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': predictions
    })
    
    # 保存
    os.makedirs('submissions', exist_ok=True)
    output_path = f'submissions/large_{model_name}_submission.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ 提交文件已保存: {output_path}")
    print(f"\n预测统计:")
    print(f"  预测为0（无糖尿病）: {(predictions == 0).sum()} ({(predictions == 0).sum() / len(predictions) * 100:.2f}%)")
    print(f"  预测为1（有糖尿病）: {(predictions == 1).sum()} ({(predictions == 1).sum() / len(predictions) * 100:.2f}%)")
    print(f"  平均预测概率: {proba.mean():.4f}")
    
    return submission


def generate_all_submissions():
    """生成所有模型的提交文件"""
    print("=" * 70)
    print("生成所有提交文件")
    print("=" * 70)
    
    models = ['ensemble', 'lightgbm', 'xgboost', 'randomforest', 'logisticregression']
    
    for model_name in models:
        try:
            print(f"\n{'-' * 70}")
            generate_submission(model_name)
        except Exception as e:
            print(f"错误 - {model_name}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("所有提交文件生成完成！")
    print("=" * 70)
    print("\n提交文件位置: submissions/")
    print("推荐使用: large_ensemble_submission.csv")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        generate_submission(model_name)
    else:
        print("\n选择生成方式:")
        print("1. 集成模型（推荐）")
        print("2. LightGBM")
        print("3. XGBoost")
        print("4. Random Forest")
        print("5. Logistic Regression")
        print("6. 生成所有提交文件")
        
        choice = input("\n请选择 (1-6，默认1): ").strip()
        
        if choice == '2':
            generate_submission('lightgbm')
        elif choice == '3':
            generate_submission('xgboost')
        elif choice == '4':
            generate_submission('randomforest')
        elif choice == '5':
            generate_submission('logisticregression')
        elif choice == '6':
            generate_all_submissions()
        else:
            generate_submission('ensemble')
    
    print("\n完成！")


if __name__ == '__main__':
    main()

