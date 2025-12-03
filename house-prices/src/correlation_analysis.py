#!/usr/bin/env python3
"""
房价预测 - 相关性分析代码
专门用于分析特征与目标变量的相关性，辅助特征工程

功能：
1. 计算特征与 SalePrice 的相关系数
2. 可视化相关性（文本和数值）
3. 识别高相关性特征对（多重共线性检测）
4. 提供特征工程建议
5. 导出相关性报告

用法:
    python src/correlation_analysis.py
    python src/correlation_analysis.py --output report.txt
    python src/correlation_analysis.py --top-n 30 --threshold 0.75
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 项目路径
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TARGET_COL = 'SalePrice'


class CorrelationAnalyzer:
    """相关性分析器"""
    
    def __init__(self, train_df: pd.DataFrame, target_col: str = TARGET_COL):
        """
        初始化分析器
        
        Args:
            train_df: 训练数据
            target_col: 目标变量列名
        """
        self.train_df = train_df.copy()
        self.target_col = target_col
        
        # 验证目标变量存在
        if target_col not in self.train_df.columns:
            raise ValueError(f"目标变量 '{target_col}' 不存在于数据中")
        
        # 获取数值型特征
        self.numeric_cols = self._get_numeric_features()
        
        # 存储分析结果
        self.target_correlations: Dict[str, float] = {}
        self.feature_correlations: List[Tuple[str, str, float]] = []
    
    def _get_numeric_features(self) -> List[str]:
        """获取数值型特征列表"""
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除目标变量和ID
        numeric_cols = [col for col in numeric_cols 
                       if col not in [self.target_col, 'Id']]
        return numeric_cols
    
    def analyze_target_correlations(self, top_n: int = 20) -> Dict[str, float]:
        """
        分析特征与目标变量的相关性
        
        Args:
            top_n: 返回前 N 个特征
            
        Returns:
            特征名到相关系数的字典（按绝对值排序）
        """
        print("\n" + "=" * 80)
        print("特征与目标变量相关性分析")
        print("=" * 80)
        print(f"目标变量: {self.target_col}")
        print(f"分析特征数: {len(self.numeric_cols)}")
        print()
        
        # 计算相关性
        correlations = {}
        for col in self.numeric_cols:
            try:
                corr = self.train_df[col].corr(self.train_df[self.target_col])
                if not pd.isna(corr):
                    correlations[col] = corr
            except Exception as e:
                print(f"警告: 计算 {col} 的相关性时出错: {e}")
        
        # 按绝对值排序
        sorted_correlations = sorted(
            correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # 存储结果
        self.target_correlations = dict(sorted_correlations)
        
        # 输出结果
        self._print_correlation_table(sorted_correlations[:top_n])
        
        return self.target_correlations
    
    def _print_correlation_table(self, correlations: List[Tuple[str, float]]):
        """打印相关性表格"""
        print(f"{'排名':<6} {'特征名称':<30} {'相关系数':>12} {'相关性强度':<20}")
        print("-" * 75)
        
        for rank, (feature, corr) in enumerate(correlations, 1):
            strength = self._get_correlation_strength(corr)
            sign = "+" if corr >= 0 else "-"
            print(f"{rank:<6} {feature:<30} {corr:>12.4f} {strength:<20}")
        
        print()
        print(f"共分析了 {len(correlations)} 个特征")
    
    def _get_correlation_strength(self, corr: float) -> str:
        """获取相关性强度描述"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "强相关 ⭐⭐⭐"
        elif abs_corr >= 0.5:
            return "中等相关 ⭐⭐"
        elif abs_corr >= 0.3:
            return "弱相关 ⭐"
        else:
            return "很弱"
    
    def analyze_feature_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        分析特征之间的相关性（检测多重共线性）
        
        Args:
            threshold: 相关性阈值，高于此值认为高度相关
            
        Returns:
            高度相关的特征对列表 [(特征1, 特征2, 相关系数), ...]
        """
        print("\n" + "=" * 80)
        print("特征之间相关性分析（多重共线性检测）")
        print("=" * 80)
        print(f"相关性阈值: {threshold}")
        print()
        
        # 计算相关性矩阵
        corr_matrix = self.train_df[self.numeric_cols].corr()
        
        # 找出高度相关的特征对
        high_corr_pairs = []
        n_features = len(corr_matrix.columns)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_val))
        
        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 存储结果
        self.feature_correlations = high_corr_pairs
        
        # 输出结果
        if high_corr_pairs:
            print(f"发现 {len(high_corr_pairs)} 对高度相关的特征（相关系数 >= {threshold}）：")
            print()
            print(f"{'特征1':<25} {'特征2':<25} {'相关系数':>12} {'建议':<20}")
            print("-" * 85)
            
            for col1, col2, corr in high_corr_pairs:
                suggestion = self._get_multicollinearity_suggestion(col1, col2)
                print(f"{col1:<25} {col2:<25} {corr:>12.4f} {suggestion:<20}")
            
            print()
            print("⚠️  多重共线性处理建议：")
            print("   1. 删除其中一个特征（保留与目标变量相关性更高的）")
            print("   2. 创建组合特征（如：平均值、差值、比例）")
            print("   3. 使用主成分分析（PCA）降维")
        else:
            print(f"✓ 未发现高度相关的特征对（阈值: {threshold}）")
        
        return high_corr_pairs
    
    def _get_multicollinearity_suggestion(self, col1: str, col2: str) -> str:
        """获取多重共线性处理建议"""
        # 比较两个特征与目标变量的相关性
        corr1 = abs(self.target_correlations.get(col1, 0))
        corr2 = abs(self.target_correlations.get(col2, 0))
        
        if corr1 > corr2:
            return f"保留 {col1}"
        elif corr2 > corr1:
            return f"保留 {col2}"
        else:
            return "创建组合特征"
    
    def get_feature_engineering_suggestions(self) -> Dict[str, List[str]]:
        """
        基于相关性分析提供特征工程建议
        
        Returns:
            建议字典，键为建议类型，值为特征列表
        """
        print("\n" + "=" * 80)
        print("特征工程建议")
        print("=" * 80)
        
        # 高相关性特征（相关系数 > 0.5）
        high_corr_features = [
            feat for feat, corr in self.target_correlations.items() 
            if abs(corr) >= 0.5
        ]
        
        suggestions = {
            'area_features': [],
            'quality_features': [],
            'time_features': [],
            'count_features': [],
            'interaction_features': []
        }
        
        # 分类特征
        for feature in high_corr_features:
            feature_lower = feature.lower()
            
            if 'sf' in feature_lower or 'area' in feature_lower:
                suggestions['area_features'].append(feature)
            elif 'qual' in feature_lower or 'cond' in feature_lower:
                suggestions['quality_features'].append(feature)
            elif 'year' in feature_lower or 'yr' in feature_lower:
                suggestions['time_features'].append(feature)
            elif 'bath' in feature_lower or 'room' in feature_lower or 'cars' in feature_lower:
                suggestions['count_features'].append(feature)
        
        # 输出建议
        self._print_suggestions(suggestions, high_corr_features)
        
        return suggestions
    
    def _print_suggestions(self, suggestions: Dict[str, List[str]], 
                          high_corr_features: List[str]):
        """打印特征工程建议"""
        
        if suggestions['area_features']:
            print("\n📐 面积相关特征（高相关性）:")
            print(f"   特征: {', '.join(suggestions['area_features'][:5])}")
            print("   建议：")
            print("   - 创建总面积特征（TotalSF = 地下室 + 一楼 + 二楼）")
            print("   - 创建面积比例特征（如：地下室占比 = TotalBsmtSF / TotalSF）")
            print("   - 考虑面积的多项式特征（平方、立方、平方根）")
            print("   - 创建面积交互特征（如：OverallQual × GrLivArea）")
        
        if suggestions['quality_features']:
            print("\n⭐ 质量相关特征（高相关性）:")
            print(f"   特征: {', '.join(suggestions['quality_features'][:5])}")
            print("   建议：")
            print("   - 创建总质量评分（TotalQual = 各质量特征之和）")
            print("   - 将质量文字转换为数值（Po=1, Fa=2, TA=3, Gd=4, Ex=5）")
            print("   - 创建质量交互特征（如：OverallQual × GrLivArea）")
        
        if suggestions['time_features']:
            print("\n📅 时间相关特征（高相关性）:")
            print(f"   特征: {', '.join(suggestions['time_features'])}")
            print("   建议：")
            print("   - 创建房屋年龄特征（HouseAge = YrSold - YearBuilt）")
            print("   - 创建改建年龄特征（RemodAge = YrSold - YearRemodAdd）")
            print("   - 创建车库年龄特征（GarageAge = YrSold - GarageYrBlt）")
        
        if suggestions['count_features']:
            print("\n🔢 数量相关特征（高相关性）:")
            print(f"   特征: {', '.join(suggestions['count_features'][:5])}")
            print("   建议：")
            print("   - 创建总浴室数（TotalBathrooms = 全浴室 + 0.5×半浴室）")
            print("   - 创建总房间数（TotalRooms = 各房间数之和）")
            print("   - 创建二元特征（HasGarage, HasBasement 等）")
        
        # 交互特征建议
        top_features = list(self.target_correlations.keys())[:5]
        if len(top_features) >= 2:
            print("\n🔗 交互特征建议：")
            print(f"   高相关性特征: {', '.join(top_features[:5])}")
            print("   建议创建交互特征：")
            for i in range(min(3, len(top_features) - 1)):
                feat1 = top_features[i]
                feat2 = top_features[i + 1]
                print(f"   - {feat1} × {feat2}")
        
        print("\n💡 通用建议：")
        print("   - 重点关注相关系数 > 0.5 的特征")
        print("   - 对于高度相关的特征对，优先创建组合特征而非删除")
        print("   - 考虑创建比例特征（如：面积比例、年龄比例）")
        print("   - 对偏态分布的特征进行 log 转换")
    
    def generate_summary_report(self) -> str:
        """生成摘要报告"""
        report = []
        report.append("=" * 80)
        report.append("相关性分析摘要报告")
        report.append("=" * 80)
        report.append(f"\n数据规模: {self.train_df.shape[0]} 行 × {self.train_df.shape[1]} 列")
        report.append(f"分析特征数: {len(self.numeric_cols)}")
        report.append(f"目标变量: {self.target_col}")
        
        # 高相关性特征统计
        strong_corr = sum(1 for c in self.target_correlations.values() if abs(c) >= 0.7)
        medium_corr = sum(1 for c in self.target_correlations.values() if 0.5 <= abs(c) < 0.7)
        weak_corr = sum(1 for c in self.target_correlations.values() if abs(c) < 0.5)
        
        report.append(f"\n相关性统计：")
        report.append(f"  强相关（≥0.7）: {strong_corr} 个特征")
        report.append(f"  中等相关（0.5-0.7）: {medium_corr} 个特征")
        report.append(f"  弱相关（<0.5）: {weak_corr} 个特征")
        
        # 多重共线性统计
        report.append(f"\n多重共线性：")
        report.append(f"  高度相关的特征对: {len(self.feature_correlations)} 对")
        
        # Top 5 特征
        top_5 = list(self.target_correlations.items())[:5]
        report.append(f"\nTop 5 重要特征：")
        for i, (feat, corr) in enumerate(top_5, 1):
            report.append(f"  {i}. {feat}: {corr:.4f}")
        
        return "\n".join(report)
    
    def save_report(self, output_path: Path):
        """保存完整报告到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # 重定向输出
            original_stdout = sys.stdout
            sys.stdout = f
            
            try:
                print("=" * 80)
                print("House Prices 相关性分析完整报告")
                print("=" * 80)
                print(f"\n生成时间: {pd.Timestamp.now()}")
                print(f"数据规模: {self.train_df.shape[0]} 行 × {self.train_df.shape[1]} 列")
                print()
                
                # 运行所有分析
                self.analyze_target_correlations(top_n=50)
                self.analyze_feature_correlations(threshold=0.7)
                self.get_feature_engineering_suggestions()
                
                print("\n" + self.generate_summary_report())
                
            finally:
                sys.stdout = original_stdout
        
        print(f"\n✓ 报告已保存到: {output_path}")


def load_data(data_path: Path) -> pd.DataFrame:
    """加载数据"""
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ 成功加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='房价预测相关性分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python src/correlation_analysis.py
  
  # 显示更多特征
  python src/correlation_analysis.py --top-n 30
  
  # 保存报告
  python src/correlation_analysis.py --output report.txt
  
  # 自定义阈值
  python src/correlation_analysis.py --threshold 0.8
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=str(DATA_DIR / 'train.csv'),
        help='训练数据路径（默认: data/train.csv）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出报告文件路径（可选）'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='显示前 N 个相关性最高的特征（默认: 20）'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='特征间相关性阈值（默认: 0.7）'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=TARGET_COL,
        help=f'目标变量列名（默认: {TARGET_COL}）'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        train_df = load_data(Path(args.data))
        
        # 创建分析器
        analyzer = CorrelationAnalyzer(train_df, target_col=args.target)
        
        # 运行分析
        analyzer.analyze_target_correlations(top_n=args.top_n)
        analyzer.analyze_feature_correlations(threshold=args.threshold)
        analyzer.get_feature_engineering_suggestions()
        
        # 保存报告（如果指定）
        if args.output:
            output_path = Path(args.output)
            analyzer.save_report(output_path)
        
        # 打印摘要
        print("\n" + analyzer.generate_summary_report())
        
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


