"""
创建示例数据集
用于测试模型训练流程
基于Pima Indians Diabetes Dataset的格式
"""

import pandas as pd
import numpy as np


def create_sample_data():
    """创建示例训练和测试数据"""
    np.random.seed(42)
    
    # 训练集样本数
    n_train = 600
    # 测试集样本数
    n_test = 168
    
    print("创建示例数据集...")
    
    # 创建训练数据
    train_data = {
        'Pregnancies': np.random.randint(0, 15, n_train),
        'Glucose': np.random.normal(120, 30, n_train).clip(0, 200),
        'BloodPressure': np.random.normal(70, 15, n_train).clip(0, 120),
        'SkinThickness': np.random.normal(20, 15, n_train).clip(0, 100),
        'Insulin': np.random.normal(80, 100, n_train).clip(0, 850),
        'BMI': np.random.normal(32, 7, n_train).clip(0, 70),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, n_train),
        'Age': np.random.randint(21, 81, n_train)
    }
    
    train_df = pd.DataFrame(train_data)
    
    # 生成目标变量（基于特征的简单规则）
    # 葡萄糖高、BMI高、年龄大的更容易患糖尿病
    risk_score = (
        (train_df['Glucose'] > 140) * 0.3 +
        (train_df['BMI'] > 30) * 0.2 +
        (train_df['Age'] > 50) * 0.15 +
        (train_df['BloodPressure'] > 80) * 0.1 +
        (train_df['Insulin'] > 150) * 0.1 +
        (train_df['Pregnancies'] > 5) * 0.05 +
        np.random.uniform(0, 0.3, n_train)
    )
    
    train_df['Outcome'] = (risk_score > 0.5).astype(int)
    
    # 随机设置一些零值（模拟真实数据的缺失）
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_columns:
        zero_indices = np.random.choice(train_df.index, size=int(n_train * 0.05), replace=False)
        train_df.loc[zero_indices, col] = 0
    
    # 创建测试数据（不包含Outcome列）
    test_data = {
        'Id': range(1, n_test + 1),
        'Pregnancies': np.random.randint(0, 15, n_test),
        'Glucose': np.random.normal(120, 30, n_test).clip(0, 200),
        'BloodPressure': np.random.normal(70, 15, n_test).clip(0, 120),
        'SkinThickness': np.random.normal(20, 15, n_test).clip(0, 100),
        'Insulin': np.random.normal(80, 100, n_test).clip(0, 850),
        'BMI': np.random.normal(32, 7, n_test).clip(0, 70),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, n_test),
        'Age': np.random.randint(21, 81, n_test)
    }
    
    test_df = pd.DataFrame(test_data)
    
    # 随机设置测试集的零值
    for col in zero_columns:
        zero_indices = np.random.choice(test_df.index, size=int(n_test * 0.05), replace=False)
        test_df.loc[zero_indices, col] = 0
    
    # 创建示例提交文件
    sample_submission = pd.DataFrame({
        'Id': range(1, n_test + 1),
        'Outcome': 0
    })
    
    # 保存数据
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    sample_submission.to_csv('data/sample_submission.csv', index=False)
    
    print(f"\n训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    print(f"\n目标变量分布:")
    print(train_df['Outcome'].value_counts())
    print(f"\n数据已保存到 data/ 目录")
    print("  - data/train.csv")
    print("  - data/test.csv")
    print("  - data/sample_submission.csv")


if __name__ == '__main__':
    create_sample_data()

