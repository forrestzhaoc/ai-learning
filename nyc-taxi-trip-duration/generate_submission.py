"""
生成NYC Taxi Trip Duration提交文件

用法:
    python generate_submission.py --model models/nyc_taxi_model.joblib --test data/test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_processing import NYCFeatureBuilder, clean_data


def load_model(model_path: Path):
    """加载保存的模型"""
    print(f"加载模型: {model_path}")
    model_dict = joblib.load(model_path)
    
    model = model_dict['model']
    preprocessor = model_dict.get('preprocessor')
    model_type = model_dict.get('model_type', 'xgb')
    use_preprocessor = model_dict.get('use_preprocessor', True)
    use_log_target = model_dict.get('use_log_target', False)
    feature_names = model_dict.get('feature_names')
    
    print(f"✓ 模型类型: {model_type}")
    print(f"✓ 使用预处理器: {use_preprocessor}")
    print(f"✓ 使用对数目标: {use_log_target}")
    
    return model, preprocessor, model_type, use_preprocessor, feature_names


def predict(model, X_test, preprocessor=None, model_type: str = 'xgb',
            use_preprocessor: bool = True, use_log_target: bool = False):
    """进行预测"""
    import numpy as np
    
    if model_type in ['xgb', 'lgb', 'rf', 'gbm', 'voting'] and use_preprocessor:
        # 树模型直接使用原始特征
        predictions_log = model.predict(X_test)
    else:
        if preprocessor is not None:
            X_test_processed = preprocessor.transform(X_test)
        else:
            X_test_processed = X_test.values
        predictions_log = model.predict(X_test_processed)
    
    # 转换回原始空间
    if use_log_target:
        predictions = np.expm1(predictions_log)  # exp(y) - 1
    else:
        predictions = predictions_log
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='生成NYC Taxi Trip Duration提交文件')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--test', type=str, required=True,
                       help='测试数据路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出提交文件路径')
    parser.add_argument('--sample-size', type=int,
                       help='采样数据大小（用于快速测试）')
    
    args = parser.parse_args()
    
    # 路径处理
    model_path = Path(args.model)
    test_path = Path(args.test)
    output_path = Path(args.output)
    
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"错误: 测试文件不存在: {test_path}")
        sys.exit(1)
    
    # 加载模型
    model, preprocessor, model_type, use_preprocessor, feature_names = load_model(model_path)
    
    # 加载测试数据
    print(f"\n加载测试数据: {test_path}")
    test_df = pd.read_csv(test_path, nrows=args.sample_size)
    print(f"✓ 测试数据: {test_df.shape[0]} 行 × {test_df.shape[1]} 列")
    
    # 保存id列
    test_ids = test_df['id'].copy() if 'id' in test_df.columns else None
    
    # 数据清洗
    print("\n数据清洗...")
    test_df = clean_data(test_df)
    
    # 特征工程
    print("特征工程...")
    feature_builder = NYCFeatureBuilder(remove_outliers=False)
    feature_builder.fit(test_df.drop(columns=['id'] if 'id' in test_df.columns else []))
    X_test = feature_builder.transform(test_df.drop(columns=['id'] if 'id' in test_df.columns else []))
    
    # 确保特征顺序一致
    if feature_names:
        missing_features = set(feature_names) - set(X_test.columns)
        extra_features = set(X_test.columns) - set(feature_names)
        
        if missing_features:
            print(f"警告: 缺少特征: {missing_features}")
            for feat in missing_features:
                X_test[feat] = 0
        
        if extra_features:
            print(f"警告: 额外特征: {extra_features}")
            X_test = X_test[feature_names]
    
    print(f"✓ 特征工程完成，特征数量: {X_test.shape[1]}")
    
    # 预测
    print("\n生成预测...")
    predictions = predict(model, X_test, preprocessor, model_type, use_preprocessor, use_log_target)
    
    # 确保预测值为正数
    predictions = np.maximum(predictions, 1.0)
    
    print(f"预测统计:")
    print(f"  均值: {predictions.mean():.2f} 秒 ({predictions.mean()/60:.2f} 分钟)")
    print(f"  中位数: {np.median(predictions):.2f} 秒 ({np.median(predictions)/60:.2f} 分钟)")
    print(f"  范围: [{predictions.min():.2f}, {predictions.max():.2f}] 秒")
    
    # 创建提交文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if test_ids is not None:
        submission_df = pd.DataFrame({
            'id': test_ids,
            'trip_duration': predictions
        })
    else:
        submission_df = pd.DataFrame({
            'trip_duration': predictions
        })
    
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ 提交文件已保存到: {output_path}")


if __name__ == "__main__":
    main()


