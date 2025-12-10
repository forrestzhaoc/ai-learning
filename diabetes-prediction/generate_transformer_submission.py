"""
使用Transformer模型生成提交文件
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from transformer_model import SimpleTabTransformer
from data_processing_transformer import TabularDataset
from torch.utils.data import DataLoader


def load_model_and_processor(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载模型和数据处理器"""
    print("=" * 70)
    print("加载Transformer模型")
    print("=" * 70)
    
    # 加载模型检查点
    checkpoint_path = 'models/transformer_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载数据处理器
    processor_path = 'models/transformer_processor.joblib'
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"数据处理器文件不存在: {processor_path}")
    
    processor = joblib.load(processor_path)
    
    # 创建模型
    model = SimpleTabTransformer(
        num_numeric_features=checkpoint['num_numeric_features'],
        categorical_cardinalities=checkpoint['categorical_cardinalities'],
        **checkpoint['model_config']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载完成")
    print(f"  最佳AUC: {checkpoint['best_auc']:.4f}")
    print(f"  设备: {device}")
    
    return model, processor


def generate_predictions(model, processor, test_df, batch_size=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """生成预测结果"""
    print("\n" + "=" * 70)
    print("生成预测结果")
    print("=" * 70)
    
    # 处理测试数据
    print("\n处理测试数据...")
    test_numeric, test_categorical, _, test_ids = processor.transform(test_df, is_test=True)
    
    print(f"测试集形状: {len(test_df)}")
    print(f"数值特征: {test_numeric.shape if test_numeric is not None else 'None'}")
    print(f"分类特征: {test_categorical.shape if test_categorical is not None else 'None'}")
    
    # 创建数据集和数据加载器
    test_dataset = TabularDataset(test_numeric, test_categorical, labels=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 生成预测
    print("\n进行预测...")
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)
            
            outputs = model(numeric_features=numeric, categorical_features=categorical)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.extend(probabilities.squeeze())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  已处理: {(batch_idx + 1) * batch_size} / {len(test_df)}")
    
    all_predictions = np.array(all_predictions)
    
    # 转换为二进制预测
    binary_predictions = (all_predictions >= 0.5).astype(int)
    
    print(f"\n预测统计:")
    print(f"  预测为0（无糖尿病）: {(binary_predictions == 0).sum()} ({(binary_predictions == 0).sum() / len(binary_predictions) * 100:.2f}%)")
    print(f"  预测为1（有糖尿病）: {(binary_predictions == 1).sum()} ({(binary_predictions == 1).sum() / len(binary_predictions) * 100:.2f}%)")
    print(f"  平均预测概率: {all_predictions.mean():.4f}")
    
    return binary_predictions, test_ids


def main():
    """主函数"""
    # 检查文件
    required_files = [
        'data/test.csv',
        'models/transformer_model.pth',
        'models/transformer_processor.joblib'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("错误：缺少必要文件:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请先运行 'python3 train_transformer.py' 训练模型。")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型和处理器
    model, processor = load_model_and_processor(device)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_df = pd.read_csv('data/test.csv')
    print(f"测试集形状: {test_df.shape}")
    
    # 生成预测
    predictions, test_ids = generate_predictions(model, processor, test_df, device=device)
    
    # 创建提交文件
    print("\n" + "=" * 70)
    print("创建提交文件")
    print("=" * 70)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': predictions
    })
    
    # 保存提交文件
    os.makedirs('submissions', exist_ok=True)
    submission_path = 'submissions/transformer_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✅ 提交文件已保存: {submission_path}")
    print(f"文件大小: {os.path.getsize(submission_path) / 1024 / 1024:.2f} MB")
    
    # 显示前几行
    print("\n提交文件预览:")
    print(submission.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n提交文件位置: {submission_path}")
    print("可以上传到Kaggle进行提交了！")


if __name__ == '__main__':
    main()





