"""
NYC Taxi Trip Duration 探索性数据分析（EDA）脚本

用法:
    python src/eda.py
    python src/eda.py --output eda_report.txt
    python src/eda.py --visualize  # 生成可视化图表
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("警告: matplotlib 或 seaborn 未安装，可视化功能将不可用")

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"


def load_data():
    """加载数据"""
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        print(f"错误: 找不到数据文件 {train_path}")
        print("请运行 python download_data.py 下载数据")
        sys.exit(1)
    
    print(f"加载数据: {train_path}")
    train_df = pd.read_csv(train_path, nrows=None)
    print(f"✓ 成功加载数据: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
    return train_df


def basic_statistics(df: pd.DataFrame, output_file=None):
    """基本统计信息"""
    output = []
    output.append("=" * 80)
    output.append("基本统计信息")
    output.append("=" * 80)
    output.append(f"\n数据形状: {df.shape[0]} 行 × {df.shape[1]} 列\n")
    
    output.append("\n数据类型:")
    output.append(df.dtypes.to_string())
    
    output.append("\n\n缺失值统计:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    output.append(missing_df[missing_df['Missing Count'] > 0].to_string())
    
    output.append("\n\n数值特征统计:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    output.append(df[numeric_cols].describe().to_string())
    
    output.append("\n\n分类特征统计:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        output.append(f"\n{col}:")
        output.append(df[col].value_counts().head(10).to_string())
    
    return "\n".join(output)


def analyze_target(df: pd.DataFrame, target_col='trip_duration'):
    """分析目标变量"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("目标变量分析")
    output.append("=" * 80)
    
    if target_col not in df.columns:
        return "\n目标变量不存在\n"
    
    target = df[target_col]
    output.append(f"\n基本统计:")
    output.append(target.describe().to_string())
    
    # 转换为分钟和小时
    output.append(f"\n\n单位转换 (示例):")
    output.append(f"最小值: {target.min()} 秒 = {target.min()/60:.2f} 分钟 = {target.min()/3600:.2f} 小时")
    output.append(f"最大值: {target.max()} 秒 = {target.max()/60:.2f} 分钟 = {target.max()/3600:.2f} 小时")
    output.append(f"平均值: {target.mean():.2f} 秒 = {target.mean()/60:.2f} 分钟 = {target.mean()/3600:.2f} 小时")
    output.append(f"中位数: {target.median():.2f} 秒 = {target.median()/60:.2f} 分钟 = {target.median()/3600:.2f} 小时")
    
    # 异常值检测（使用IQR方法）
    Q1 = target.quantile(0.25)
    Q3 = target.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((target < lower_bound) | (target > upper_bound)).sum()
    
    output.append(f"\n\n异常值检测 (IQR方法):")
    output.append(f"Q1: {Q1:.2f} 秒")
    output.append(f"Q3: {Q3:.2f} 秒")
    output.append(f"IQR: {IQR:.2f} 秒")
    output.append(f"下界: {lower_bound:.2f} 秒")
    output.append(f"上界: {upper_bound:.2f} 秒")
    output.append(f"异常值数量: {outliers} ({outliers/len(target)*100:.2f}%)")
    
    return "\n".join(output)


def analyze_geospatial(df: pd.DataFrame):
    """分析地理空间特征"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("地理空间特征分析")
    output.append("=" * 80)
    
    geo_cols = ['pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude']
    
    for col in geo_cols:
        if col in df.columns:
            output.append(f"\n{col}:")
            output.append(f"  范围: [{df[col].min():.6f}, {df[col].max():.6f}]")
            output.append(f"  均值: {df[col].mean():.6f}")
            output.append(f"  标准差: {df[col].std():.6f}")
    
    # 计算距离（如果可能）
    if all(col in df.columns for col in geo_cols):
        # 使用Haversine距离公式的简化版本（曼哈顿距离作为近似）
        df_temp = df.head(1000).copy()  # 采样以避免计算太慢
        df_temp['lat_diff'] = np.abs(df_temp['dropoff_latitude'] - df_temp['pickup_latitude'])
        df_temp['lon_diff'] = np.abs(df_temp['dropoff_longitude'] - df_temp['pickup_longitude'])
        df_temp['manhattan_dist'] = df_temp['lat_diff'] + df_temp['lon_diff']
        
        output.append(f"\n\n采样距离分析 (前1000行):")
        output.append(f"平均曼哈顿距离: {df_temp['manhattan_dist'].mean():.6f}")
        output.append(f"最大曼哈顿距离: {df_temp['manhattan_dist'].max():.6f}")
        output.append(f"最小曼哈顿距离: {df_temp['manhattan_dist'].min():.6f}")
    
    return "\n".join(output)


def analyze_datetime(df: pd.DataFrame):
    """分析日期时间特征"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("日期时间特征分析")
    output.append("=" * 80)
    
    if 'pickup_datetime' in df.columns:
        df_temp = df.copy()
        df_temp['pickup_datetime'] = pd.to_datetime(df_temp['pickup_datetime'])
        
        output.append(f"\n时间范围:")
        output.append(f"最早: {df_temp['pickup_datetime'].min()}")
        output.append(f"最晚: {df_temp['pickup_datetime'].max()}")
        output.append(f"跨度: {df_temp['pickup_datetime'].max() - df_temp['pickup_datetime'].min()}")
        
        df_temp['hour'] = df_temp['pickup_datetime'].dt.hour
        df_temp['day_of_week'] = df_temp['pickup_datetime'].dt.dayofweek
        df_temp['month'] = df_temp['pickup_datetime'].dt.month
        
        output.append(f"\n\n小时分布 (top 5):")
        output.append(df_temp['hour'].value_counts().head().to_string())
        
        output.append(f"\n\n星期分布:")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df_temp['day_of_week'].value_counts().sort_index()
        for day, count in zip(day_names, day_counts):
            output.append(f"{day}: {count}")
    
    return "\n".join(output)


def correlation_analysis(df: pd.DataFrame):
    """相关性分析"""
    output = []
    output.append("\n" + "=" * 80)
    output.append("特征-目标相关性分析")
    output.append("=" * 80)
    
    # 特征工程（简化版，用于EDA）
    df_eng = engineer_features_for_eda(df.copy())
    
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
    if 'trip_duration' in numeric_cols:
        numeric_cols.remove('trip_duration')
    
    if 'trip_duration' in df_eng.columns and len(numeric_cols) > 0:
        correlations = df_eng[numeric_cols + ['trip_duration']].corr()['trip_duration'].abs().sort_values(ascending=False)
        correlations = correlations[correlations.index != 'trip_duration']
        
        output.append(f"\n与 trip_duration 的相关性 (绝对值):")
        output.append(correlations.head(20).to_string())
    
    return "\n".join(output)


def engineer_features_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程（用于EDA分析）"""
    df = df.copy()
    
    # 日期时间特征
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['day_of_month'] = df['pickup_datetime'].dt.day
        df['month'] = df['pickup_datetime'].dt.month
        df['year'] = df['pickup_datetime'].dt.year
    
    # 地理距离特征（简化版）
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                          'dropoff_latitude', 'dropoff_longitude']):
        df['lat_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
        df['lon_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
        df['manhattan_distance'] = np.abs(df['lat_diff']) + np.abs(df['lon_diff'])
        
        # Haversine距离（使用简化的球面距离）
        R = 6371  # 地球半径（公里）
        df['pickup_lat_rad'] = np.radians(df['pickup_latitude'])
        df['dropoff_lat_rad'] = np.radians(df['dropoff_latitude'])
        df['delta_lat'] = np.radians(df['dropoff_latitude'] - df['pickup_latitude'])
        df['delta_lon'] = np.radians(df['dropoff_longitude'] - df['pickup_longitude'])
        
        a = np.sin(df['delta_lat']/2)**2 + \
            np.cos(df['pickup_lat_rad']) * np.cos(df['dropoff_lat_rad']) * \
            np.sin(df['delta_lon']/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        df['haversine_distance'] = R * c
    
    return df


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """创建可视化图表"""
    if not HAS_VISUALIZATION:
        print("可视化库未安装，跳过图表生成")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 目标变量分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df['trip_duration'].hist(bins=50, edgecolor='black')
    plt.title('Trip Duration Distribution')
    plt.xlabel('Trip Duration (seconds)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df['trip_duration'].apply(lambda x: np.log1p(x)).hist(bins=50, edgecolor='black')
    plt.title('Log Trip Duration Distribution')
    plt.xlabel('Log(Trip Duration)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 地理分布（采样）
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude']):
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(df_sample['pickup_longitude'], df_sample['pickup_latitude'], 
                   alpha=0.1, s=1)
        plt.title(f'Pickup Locations (sample of {sample_size})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(output_dir / 'pickup_locations.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 时间特征分析
    if 'pickup_datetime' in df.columns:
        df_temp = df.copy()
        df_temp['pickup_datetime'] = pd.to_datetime(df_temp['pickup_datetime'])
        df_temp['hour'] = df_temp['pickup_datetime'].dt.hour
        df_temp['day_of_week'] = df_temp['pickup_datetime'].dt.dayofweek
        
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        df_temp.groupby('hour')['trip_duration'].mean().plot(kind='bar')
        plt.title('Average Trip Duration by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Duration (seconds)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df_temp.groupby('day_of_week')['trip_duration'].mean().plot(kind='bar')
        plt.title('Average Trip Duration by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Duration (seconds)')
        plt.xticks(range(7), day_names, rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'time_features.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 可视化图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='NYC Taxi Trip Duration EDA')
    parser.add_argument('--output', type=str, help='输出报告文件路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    args = parser.parse_args()
    
    # 加载数据
    df = load_data()
    
    # 执行分析
    report_parts = []
    report_parts.append(basic_statistics(df))
    report_parts.append(analyze_target(df))
    report_parts.append(analyze_geospatial(df))
    report_parts.append(analyze_datetime(df))
    report_parts.append(correlation_analysis(df))
    
    full_report = "\n".join(report_parts)
    
    # 输出报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\n✓ 报告已保存到: {output_path}")
    else:
        print(full_report)
    
    # 生成可视化
    if args.visualize:
        viz_dir = ROOT_DIR / "eda_plots"
        create_visualizations(df, viz_dir)


if __name__ == "__main__":
    main()


