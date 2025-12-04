"""
探索性数据分析 (EDA)
分析糖尿病预测数据集的统计特征和分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(data_path='data/train.csv'):
    """加载训练数据"""
    return pd.read_csv(data_path)


def basic_info(df):
    """打印基本信息"""
    print("=" * 60)
    print("数据集基本信息")
    print("=" * 60)
    print(f"数据形状: {df.shape}")
    print(f"\n列名:\n{df.columns.tolist()}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    print(f"\n基本统计信息:\n{df.describe()}")
    
    # 检查目标变量分布
    if 'Outcome' in df.columns:
        print(f"\n目标变量分布:")
        print(df['Outcome'].value_counts())
        print(f"\n目标变量比例:")
        print(df['Outcome'].value_counts(normalize=True))


def check_zero_values(df):
    """检查零值（某些特征不应该为0）"""
    print("\n" + "=" * 60)
    print("零值检查")
    print("=" * 60)
    
    # 这些特征理论上不应该为0
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI']
    
    for column in zero_not_accepted:
        if column in df.columns:
            zero_count = (df[column] == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
            print(f"{column}: {zero_count} 个零值 ({zero_percentage:.2f}%)")


def plot_distributions(df, save_path='predictions'):
    """绘制特征分布图"""
    print("\n" + "=" * 60)
    print("生成分布图...")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 获取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Outcome' in numeric_cols:
        numeric_cols.remove('Outcome')
    
    # 绘制直方图
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    plt.figure(figsize=(15, 5 * n_rows))
    for idx, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, 3, idx)
        plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'{col} 分布')
        plt.xlabel(col)
        plt.ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"分布图已保存到: {save_path}/feature_distributions.png")
    plt.close()


def plot_correlation_matrix(df, save_path='predictions'):
    """绘制相关性矩阵"""
    print("\n生成相关性矩阵...")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 计算相关性
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    # 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('特征相关性矩阵', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"相关性矩阵已保存到: {save_path}/correlation_matrix.png")
    plt.close()
    
    # 打印与目标变量的相关性
    if 'Outcome' in correlation_matrix.columns:
        print("\n与目标变量的相关性（绝对值排序）:")
        target_corr = correlation_matrix['Outcome'].drop('Outcome').abs().sort_values(ascending=False)
        for feature, corr in target_corr.items():
            print(f"{feature}: {correlation_matrix['Outcome'][feature]:.4f}")


def plot_box_plots(df, save_path='predictions'):
    """绘制箱线图检测异常值"""
    print("\n生成箱线图...")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 获取数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Outcome' in numeric_cols:
        numeric_cols.remove('Outcome')
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    plt.figure(figsize=(15, 5 * n_rows))
    for idx, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, 3, idx)
        plt.boxplot(df[col].dropna())
        plt.title(f'{col} 箱线图')
        plt.ylabel(col)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/box_plots.png', dpi=300, bbox_inches='tight')
    print(f"箱线图已保存到: {save_path}/box_plots.png")
    plt.close()


def main():
    """主函数"""
    # 加载数据
    print("加载数据...")
    df = load_data()
    
    # 基本信息
    basic_info(df)
    
    # 检查零值
    check_zero_values(df)
    
    # 绘制分布图
    plot_distributions(df)
    
    # 绘制相关性矩阵
    plot_correlation_matrix(df)
    
    # 绘制箱线图
    plot_box_plots(df)
    
    print("\n" + "=" * 60)
    print("EDA完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

