#!/usr/bin/env python3
"""
Titanic 探索性数据分析（EDA）脚本
用于分析特征相关性，辅助特征工程

用法:
    python src/eda.py
    python src/eda.py --output correlation_report.txt
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

from sklearn.preprocessing import LabelEncoder

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


def engineer_features_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程（用于EDA分析）
    与 train_simple.py 中的特征工程保持一致
    """
    out = df.copy()
    
    # 提取 Title
    titles = out["Name"].str.extract(r",\s*([^.]+)\.\s*", expand=False).str.strip()
    out["Title"] = titles.replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Dr": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
    }).fillna("Rare")
    
    # 填充缺失值
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["Fare"] = out["Fare"].fillna(out["Fare"].median())
    out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode()[0])
    out["Cabin"] = out["Cabin"].fillna("Unknown")
    
    # 家庭特征
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    
    # Age binning
    age_bins = [-1, 12, 18, 35, 55, 120]
    age_labels = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
        right=True,
    ).astype(str)
    
    # Fare 特征
    out["FarePerPerson"] = (out["Fare"] / out["FamilySize"]).replace([np.inf, -np.inf], np.nan)
    out["FarePerPerson"] = out["FarePerPerson"].fillna(out["Fare"])
    fare_bins = pd.qcut(out["Fare"], q=4, labels=False, duplicates="drop")
    out["FareBin"] = fare_bins.astype(float).fillna(-1).astype(int)
    
    # Cabin 特征
    out["Deck"] = out["Cabin"].str[0].fillna("U")
    out["HasCabin"] = (out["Cabin"] != "Unknown").astype(int)
    out["CabinCount"] = out["Cabin"].str.split().str.len().fillna(0).astype(int)
    
    # Ticket 特征
    out["TicketPrefix"] = (
        out["Ticket"].str.replace(r"[0-9./ ]", "", regex=True).str.upper().replace("", "NONE")
    )
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("size")
    
    # Sex 编码
    out["Sex"] = out["Sex"].map({"male": 0, "female": 1}).astype(int)
    
    return out


def analyze_target_correlations(train_df, target_col='Survived', top_n=20):
    """
    分析特征与目标变量的相关性
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
    """
    print("\n" + "=" * 80)
    print("1. 特征与目标变量 (Survived) 的相关性分析")
    print("=" * 80)
    
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in numeric_cols:
        print(f"错误: 目标变量 '{target_col}' 不是数值型")
        return None
    
    # 移除目标变量和ID
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'PassengerId']]
    
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
        if abs_corr >= 0.5:
            strength = "强相关 ⭐⭐⭐"
        elif abs_corr >= 0.3:
            strength = "中等相关 ⭐⭐"
        elif abs_corr >= 0.2:
            strength = "弱相关 ⭐"
        else:
            strength = "很弱"
        
        sign = "+" if corr > 0 else "-"
        print(f"{feature:<30} {corr:>12.4f} {strength:<15}")
    
    print(f"\n共分析了 {len(correlations)} 个数值特征")
    print(f"显示前 {min(top_n, len(correlations))} 个特征")
    
    return dict(correlations_sorted)


def analyze_feature_correlations(train_df, target_col='Survived', threshold=0.7):
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
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'PassengerId']]
    
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


def analyze_categorical_correlations(train_df, target_col='Survived', top_n=15):
    """
    分析分类特征与目标变量的关系
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
    """
    print("\n" + "=" * 80)
    print("3. 分类特征与目标变量的关系分析")
    print("=" * 80)
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Name', 'Ticket', 'Cabin']]
    
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


def analyze_survival_by_category(train_df, target_col='Survived'):
    """
    分析不同类别特征的生存率
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
    """
    print("\n" + "=" * 80)
    print("4. 分类特征生存率分析")
    print("=" * 80)
    
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Name', 'Ticket', 'Cabin']]
    
    for col in categorical_cols[:10]:  # 分析前10个分类特征
        if col in train_df.columns:
            survival_rate = train_df.groupby(col)[target_col].agg(['mean', 'count'])
            survival_rate.columns = ['生存率', '数量']
            survival_rate = survival_rate.sort_values('生存率', ascending=False)
            
            print(f"\n{col} 的生存率分析:")
            print("-" * 60)
            print(f"{'类别':<20} {'生存率':>12} {'数量':>10}")
            for category, row in survival_rate.head(10).iterrows():
                print(f"{str(category):<20} {row['生存率']:>12.2%} {int(row['数量']):>10}")


def plot_correlation_heatmap(train_df, target_col='Survived', top_n=20, save_path=None):
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
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'PassengerId']]
    
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


def plot_top_correlations(train_df, target_col='Survived', top_n=15, save_path=None):
    """
    绘制与目标变量相关性最高的特征条形图
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        top_n: 显示前 N 个特征
        save_path: 保存路径
    """
    if not HAS_VISUALIZATION:
        print("警告: 可视化库未安装，跳过条形图生成")
        return
    
    # 获取数值型特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in [target_col, 'PassengerId']]
    
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


def plot_survival_by_category(train_df, target_col='Survived', save_path=None):
    """
    绘制不同类别的生存率对比图
    
    Args:
        train_df: 训练数据
        target_col: 目标变量列名
        save_path: 保存路径
    """
    if not HAS_VISUALIZATION:
        print("警告: 可视化库未安装，跳过生存率图生成")
        return
    
    # 选择几个重要的分类特征
    important_cats = ['Sex', 'Pclass', 'Embarked', 'Title', 'AgeBin', 'Deck']
    available_cats = [col for col in important_cats if col in train_df.columns]
    
    if not available_cats:
        return
    
    n_cols = 3
    n_rows = (len(available_cats) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if len(available_cats) > 1 else [axes]
    
    for idx, col in enumerate(available_cats):
        ax = axes[idx]
        survival_rate = train_df.groupby(col)[target_col].mean().sort_values(ascending=False)
        
        bars = ax.bar(range(len(survival_rate)), survival_rate.values, alpha=0.7)
        ax.set_xticks(range(len(survival_rate)))
        ax.set_xticklabels(survival_rate.index, rotation=45, ha='right')
        ax.set_ylabel('Survival Rate', fontsize=10)
        ax.set_title(f'{col} Survival Rate', fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, survival_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(available_cats), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Survival Rate by Category', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = ROOT_DIR / "survival_by_category.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 生存率对比图已保存到: {save_path}")


def save_correlation_report(train_df, output_path=None, target_col='Survived', visualize=False):
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
            print("Titanic 特征相关性分析报告")
            print("=" * 80)
            print(f"\n生成时间: {pd.Timestamp.now()}")
            print(f"数据规模: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
            
            # 运行所有分析
            correlations = analyze_target_correlations(train_df, target_col, top_n=30)
            analyze_feature_correlations(train_df, target_col)
            analyze_categorical_correlations(train_df, target_col)
            analyze_survival_by_category(train_df, target_col)
            
        sys.stdout = original_stdout
        print(f"\n✓ 报告已保存到: {output_path}")
        
        # 生成可视化图表
        if visualize:
            print("\n生成可视化图表...")
            plot_correlation_heatmap(train_df, target_col, top_n=20)
            plot_top_correlations(train_df, target_col, top_n=15)
            plot_survival_by_category(train_df, target_col)
            print("✓ 所有可视化图表已生成")
        
    except Exception as e:
        sys.stdout = original_stdout
        print(f"保存报告时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Titanic EDA 和相关性分析')
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
    
    # 进行特征工程（与训练脚本保持一致）
    print("\n进行特征工程...")
    train_df = engineer_features_for_eda(train_df)
    
    # 运行分析
    correlations = analyze_target_correlations(train_df, top_n=args.top_n)
    analyze_feature_correlations(train_df, threshold=args.threshold)
    analyze_categorical_correlations(train_df, top_n=args.top_n)
    analyze_survival_by_category(train_df)
    
    # 保存报告（如果指定）
    if args.output:
        save_correlation_report(train_df, args.output, visualize=args.visualize)
    elif args.visualize:
        # 即使不保存报告，也生成可视化
        print("\n生成可视化图表...")
        plot_correlation_heatmap(train_df, top_n=args.top_n)
        plot_top_correlations(train_df, top_n=args.top_n)
        plot_survival_by_category(train_df)
        print("✓ 所有可视化图表已生成")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n使用建议：")
    print("1. 重点关注相关系数 > 0.3 的特征（对二分类问题，0.3 已经是较强的相关性）")
    print("2. 对于高度相关的特征对，考虑特征工程（组合或删除）")
    print("3. 根据相关性分析结果指导特征工程方向")
    print("4. 使用 --visualize 参数生成可视化图表")
    print("=" * 80)


if __name__ == '__main__':
    main()









