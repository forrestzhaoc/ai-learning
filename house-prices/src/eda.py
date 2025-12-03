#!/usr/bin/env python3
"""
探索性数据分析（EDA）脚本
用于分析特征相关性，辅助特征工程

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

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    LabelEncoder = None

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"


def load_data():
    """加载数据"""
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        print(f"错误: 找不到数据文件 {train_path}")
        print("请确保 data/train.csv 文件存在")
        sys.exit(1)
    
    train_df = pd.read_csv(train_path)
    print(f"✓ 成功加载数据: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
    return train_df


def analyze_target_correlations(train_df, target_col='SalePrice', top_n=20):
    """
    分析特征与目标变量的相关性
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
    """
    print("\n" + "=" * 80)
    print("1. 特征与目标变量 (SalePrice) 的相关性分析")
    print("=" * 80)
    
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in numeric_cols:
        print(f"错误: 目标变量 '{target_col}' 不是数值型")
        return None
    
    # 移除目标变量和ID
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'Id']]
    
    # 计算相关性
    correlations = {}
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[target_col])
        if not pd.isna(corr):
            correlations[col] = corr
    
    # 按绝对值排序
    correlations_sorted = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # 输出结果
    print(f"\n{'特征名称':<30} {'相关系数':>12} {'相关性强度':<15}")
    print("-" * 60)
    
    for feature, corr in correlations_sorted[:top_n]:
        # 判断相关性强度
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "强相关 ⭐⭐⭐"
        elif abs_corr >= 0.5:
            strength = "中等相关 ⭐⭐"
        elif abs_corr >= 0.3:
            strength = "弱相关 ⭐"
        else:
            strength = "很弱"
        
        sign = "+" if corr > 0 else "-"
        print(f"{feature:<30} {corr:>12.4f} {strength:<15}")
    
    print(f"\n共分析了 {len(correlations)} 个数值特征")
    print(f"显示前 {min(top_n, len(correlations))} 个特征")
    
    return dict(correlations_sorted)


def analyze_feature_correlations(train_df, target_col='SalePrice', threshold=0.7):
    """
    分析特征之间的相关性（检测多重共线性）
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        threshold: 相关性阈值（高于此值认为高度相关）
    """
    print("\n" + "=" * 80)
    print("2. 特征之间的相关性分析（检测多重共线性）")
    print("=" * 80)
    
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'Id']]
    
    # 计算相关性矩阵
    corr_matrix = train_df[numeric_cols].corr()
    
    # 找出高度相关的特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_val))
    
    # 按相关性绝对值排序
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if high_corr_pairs:
        print(f"\n发现 {len(high_corr_pairs)} 对高度相关的特征（相关系数 >= {threshold}）：")
        print(f"\n{'特征1':<25} {'特征2':<25} {'相关系数':>12}")
        print("-" * 65)
        for col1, col2, corr in high_corr_pairs[:20]:  # 显示前20对
            print(f"{col1:<25} {col2:<25} {corr:>12.4f}")
        
        print(f"\n⚠️  建议：高度相关的特征可能存在多重共线性问题")
        print("   可以考虑：")
        print("   1. 删除其中一个特征")
        print("   2. 创建组合特征")
        print("   3. 使用主成分分析（PCA）")
    else:
        print(f"\n✓ 未发现高度相关的特征对（阈值: {threshold}）")
    
    return high_corr_pairs


def analyze_missing_correlations(train_df, target_col='SalePrice'):
    """
    分析缺失值与目标变量的关系
    """
    print("\n" + "=" * 80)
    print("3. 缺失值分析")
    print("=" * 80)
    
    missing_data = train_df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        print(f"\n发现 {len(missing_data)} 个特征有缺失值：")
        print(f"\n{'特征名称':<30} {'缺失数量':>12} {'缺失比例':>12}")
        print("-" * 55)
        
        for feature, count in missing_data.items():
            pct = (count / len(train_df)) * 100
            print(f"{feature:<30} {count:>12} {pct:>11.2f}%")
        
        # 分析缺失值是否与目标变量相关
        print("\n分析缺失值模式与目标变量的关系：")
        print("-" * 55)
        
        for feature in missing_data.head(10).index:
            if feature != target_col:
                # 比较有缺失值和无缺失值的价格差异
                missing_mask = train_df[feature].isnull()
                if missing_mask.sum() > 0:
                    price_with_missing = train_df.loc[missing_mask, target_col].mean()
                    price_without_missing = train_df.loc[~missing_mask, target_col].mean()
                    
                    if not pd.isna(price_with_missing) and not pd.isna(price_without_missing):
                        diff = price_with_missing - price_without_missing
                        print(f"{feature:<30} 缺失时平均价格: ${price_without_missing:,.0f}, "
                              f"有值时: ${price_without_missing:,.0f}, "
                              f"差异: ${diff:,.0f}")
    else:
        print("\n✓ 未发现缺失值")


def analyze_feature_importance_for_engineering(train_df, target_col='SalePrice'):
    """
    为特征工程提供建议
    """
    print("\n" + "=" * 80)
    print("4. 特征工程建议")
    print("=" * 80)
    
    # 获取高相关性特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'Id']]
    
    correlations = {}
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[target_col])
        if not pd.isna(corr):
            correlations[col] = abs(corr)
    
    # 找出高相关性特征
    high_corr_features = [col for col, corr in correlations.items() if abs(corr) >= 0.5]
    
    print("\n基于相关性分析的特征工程建议：")
    print("-" * 80)
    
    # 面积相关特征
    area_features = [col for col in high_corr_features if 'SF' in col or 'Area' in col]
    if area_features:
        print(f"\n📐 面积相关特征（高相关性）: {', '.join(area_features[:5])}")
        print("   建议：")
        print("   - 创建总面积特征（TotalSF = 地下室 + 一楼 + 二楼）")
        print("   - 创建面积比例特征（如：地下室占比）")
        print("   - 考虑面积的多项式特征（平方、立方）")
    
    # 质量相关特征
    qual_features = [col for col in high_corr_features if 'Qual' in col or 'Cond' in col]
    if qual_features:
        print(f"\n⭐ 质量相关特征（高相关性）: {', '.join(qual_features[:5])}")
        print("   建议：")
        print("   - 创建总质量评分（TotalQual = 各质量特征之和）")
        print("   - 将质量文字转换为数值（Po=1, Fa=2, TA=3, Gd=4, Ex=5）")
    
    # 时间相关特征
    year_features = [col for col in high_corr_features if 'Year' in col or 'Yr' in col]
    if year_features:
        print(f"\n📅 时间相关特征（高相关性）: {', '.join(year_features)}")
        print("   建议：")
        print("   - 创建房屋年龄特征（HouseAge = YrSold - YearBuilt）")
        print("   - 创建改建年龄特征（RemodAge = YrSold - YearRemodAdd）")
    
    # 二元特征建议
    print(f"\n🔘 二元特征建议：")
    print("   - 对于有/无类型的特征，创建二元特征（HasGarage, HasBasement 等）")
    print("   - 这些特征可能捕捉到非线性关系")
    
    # 交互特征建议
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
    if len(top_features) >= 2:
        print(f"\n🔗 交互特征建议：")
        print(f"   - 考虑创建交互特征（如：{top_features[0][0]} × {top_features[1][0]}）")
        print("   - 交互特征可能捕捉到特征之间的协同效应")


def plot_correlation_heatmap(train_df, target_col='SalePrice', top_n=20, save_path=None):
    """
    绘制特征相关性热力图
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
        save_path: 保存路径
    """
    if not HAS_VISUALIZATION:
        print("警告: 可视化库未安装，跳过热力图生成")
        return
    
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'Id']]
    
    # 计算相关性
    correlations = {}
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[target_col])
        if not pd.isna(corr):
            correlations[col] = corr
    
    # 按绝对值排序，取前 top_n
    correlations_sorted = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [feat for feat, _ in correlations_sorted[:top_n]]
    
    # 计算这些特征之间的相关性矩阵
    corr_matrix = train_df[top_features + [target_col]].corr()
    
    # 绘制热力图
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1
    )
    plt.title(f'Top {top_n} Features Correlation Heatmap with {target_col}', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path is None:
        save_path = ROOT_DIR / "correlation_heatmap.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 热力图已保存到: {save_path}")


def plot_top_correlations(train_df, target_col='SalePrice', top_n=15, save_path=None):
    """
    绘制与目标变量相关性最高的特征条形图
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
        save_path: 保存路径
    """
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'Id']]
    
    # 计算相关性
    correlations = {}
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[target_col])
        if not pd.isna(corr):
            correlations[col] = corr
    
    # 按绝对值排序，取前 top_n
    correlations_sorted = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [feat for feat, _ in correlations_sorted[:top_n]]
    top_corrs = [corr for _, corr in correlations_sorted[:top_n]]
    
    # 绘制条形图
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'blue' for x in top_corrs]
    bars = plt.barh(range(len(top_features)), top_corrs, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.title(f'Top {top_n} Features Correlated with {target_col}', fontsize=14, pad=20)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, corr) in enumerate(zip(bars, top_corrs)):
        plt.text(corr + (0.02 if corr >= 0 else -0.02), i, f'{corr:.3f}',
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = ROOT_DIR / "top_correlations.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 相关性条形图已保存到: {save_path}")


def plot_scatter_top_features(train_df, target_col='SalePrice', top_n=6, save_path=None):
    """
    绘制与目标变量相关性最高的特征的散点图
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
        save_path: 保存路径
    """
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'Id']]
    
    # 计算相关性
    correlations = {}
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[target_col])
        if not pd.isna(corr):
            correlations[col] = corr
    
    # 按绝对值排序，取前 top_n
    correlations_sorted = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [feat for feat, _ in correlations_sorted[:top_n]]
    
    # 创建子图
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if top_n > 1 else [axes]
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        corr = correlations[feature]
        
        # 绘制散点图
        ax.scatter(train_df[feature], train_df[target_col], alpha=0.5, s=20)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel(target_col, fontsize=10)
        ax.set_title(f'{feature}\nCorr: {corr:.3f}', fontsize=11)
        ax.grid(alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(top_n, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Scatter Plots: Top {top_n} Features vs {target_col}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = ROOT_DIR / "scatter_top_features.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 散点图已保存到: {save_path}")


def analyze_categorical_correlations(train_df, target_col='SalePrice', top_n=15):
    """
    分析分类特征与目标变量的关系（通过编码后计算相关性）
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
    """
    print("\n" + "=" * 80)
    print("5. 分类特征与目标变量的关系分析")
    print("=" * 80)
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("\n未发现分类特征")
        return
    
    correlations = {}
    
    for col in categorical_cols:
        # 使用 LabelEncoder 编码
        le = LabelEncoder()
        try:
            encoded = le.fit_transform(train_df[col].fillna('Missing'))
            corr = np.corrcoef(encoded, train_df[target_col])[0, 1]
            if not pd.isna(corr):
                correlations[col] = abs(corr)
        except:
            continue
    
    if correlations:
        correlations_sorted = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'特征名称':<30} {'相关系数':>12} {'相关性强度':<15}")
        print("-" * 60)
        
        for feature, corr in correlations_sorted[:top_n]:
            abs_corr = abs(corr)
            if abs_corr >= 0.5:
                strength = "中等相关 ⭐⭐"
            elif abs_corr >= 0.3:
                strength = "弱相关 ⭐"
            else:
                strength = "很弱"
            
            print(f"{feature:<30} {corr:>12.4f} {strength:<15}")
        
        print(f"\n共分析了 {len(correlations)} 个分类特征")
    else:
        print("\n无法计算分类特征相关性")


def save_correlation_report(train_df, output_path=None, target_col='SalePrice', visualize=False):
    """
    保存相关性分析报告
    
    Args:
        train_df: 训练数据
        output_path: 输出文件路径
        target_col: 目标变量列名
        visualize: 是否生成可视化图表
    """
    if output_path is None:
        output_path = ROOT_DIR / "correlation_report.txt"
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            print("=" * 80)
            print("House Prices 特征相关性分析报告")
            print("=" * 80)
            print(f"\n生成时间: {pd.Timestamp.now()}")
            print(f"数据规模: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
            
            # 运行所有分析
            correlations = analyze_target_correlations(train_df, target_col, top_n=30)
            analyze_feature_correlations(train_df, target_col)
            analyze_missing_correlations(train_df, target_col)
            analyze_feature_importance_for_engineering(train_df, target_col)
            analyze_categorical_correlations(train_df, target_col)
            
        sys.stdout = original_stdout
        print(f"\n✓ 报告已保存到: {output_path}")
        
        # 生成可视化图表
        if visualize:
            print("\n生成可视化图表...")
            plot_correlation_heatmap(train_df, target_col, top_n=20)
            plot_top_correlations(train_df, target_col, top_n=15)
            plot_scatter_top_features(train_df, target_col, top_n=6)
            print("✓ 所有可视化图表已生成")
        
    except Exception as e:
        sys.stdout = original_stdout
        print(f"保存报告时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='EDA 和相关性分析')
    parser.add_argument('--data', type=str, default=str(DATA_DIR / 'train.csv'),
                       help='训练数据路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径（可选）')
    parser.add_argument('--top-n', type=int, default=20,
                       help='显示前 N 个相关性最高的特征')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='特征间相关性阈值')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    
    args = parser.parse_args()
    
    # 加载数据
    train_df = load_data()
    
    # 运行分析
    correlations = analyze_target_correlations(train_df, top_n=args.top_n)
    analyze_feature_correlations(train_df, threshold=args.threshold)
    analyze_missing_correlations(train_df)
    analyze_feature_importance_for_engineering(train_df)
    analyze_categorical_correlations(train_df, top_n=args.top_n)
    
    # 保存报告（如果指定）
    if args.output:
        save_correlation_report(train_df, args.output, visualize=args.visualize)
    elif args.visualize:
        # 即使不保存报告，也生成可视化
        print("\n生成可视化图表...")
        plot_correlation_heatmap(train_df, top_n=args.top_n)
        plot_top_correlations(train_df, top_n=args.top_n)
        plot_scatter_top_features(train_df, top_n=min(6, args.top_n))
        print("✓ 所有可视化图表已生成")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n使用建议：")
    print("1. 重点关注相关系数 > 0.5 的特征")
    print("2. 对于高度相关的特征对，考虑特征工程（组合或删除）")
    print("3. 根据相关性分析结果指导特征工程方向")
    print("4. 使用 --visualize 参数生成可视化图表")
    print("=" * 80)


if __name__ == '__main__':
    main()

