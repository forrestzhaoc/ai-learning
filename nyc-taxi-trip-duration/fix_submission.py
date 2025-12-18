#!/usr/bin/env python3
"""
修复提交文件 - 确保符合Kaggle要求

用法:
    python fix_submission.py --submission predictions/submission.csv --test data/test.csv
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def fix_submission(submission_path: Path, test_path: Path, output_path: Path | None = None):
    """修复提交文件，确保包含所有测试集ID"""
    print("修复提交文件...")
    print(f"  提交文件: {submission_path}")
    print(f"  测试集: {test_path}")
    
    # 读取文件
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(submission_path)
    
    print(f"\n原始状态:")
    print(f"  测试集ID数量: {len(test_df)}")
    print(f"  提交文件ID数量: {len(sub_df)}")
    
    # 找出缺失的ID
    missing_ids = set(test_df['id']) - set(sub_df['id'])
    extra_ids = set(sub_df['id']) - set(test_df['id'])
    
    print(f"  缺失的ID数量: {len(missing_ids)}")
    print(f"  多余的ID数量: {len(extra_ids)}")
    
    if len(missing_ids) == 0 and len(extra_ids) == 0:
        # 只需要确保顺序和格式正确
        print("\n✓ 所有ID都存在，只需确保顺序正确...")
        sub_df = sub_df.set_index('id').reindex(test_df['id']).reset_index()
    else:
        # 需要修复
        if len(missing_ids) > 0:
            # 使用中位数填充缺失的ID
            median_duration = sub_df['trip_duration'].median()
            print(f"\n使用中位数填充 {len(missing_ids)} 个缺失ID: {median_duration:.2f} 秒")
            
            missing_df = pd.DataFrame({
                'id': list(missing_ids),
                'trip_duration': median_duration
            })
            
            # 合并
            sub_df = pd.concat([sub_df, missing_df], ignore_index=True)
        
        if len(extra_ids) > 0:
            # 移除多余的ID
            print(f"\n移除 {len(extra_ids)} 个多余的ID")
            sub_df = sub_df[sub_df['id'].isin(test_df['id'])]
        
        # 按照测试集的顺序排序
        sub_df = sub_df.set_index('id').reindex(test_df['id']).reset_index()
    
    # 确保预测值合理
    sub_df['trip_duration'] = np.maximum(sub_df['trip_duration'], 1.0)
    
    # 保存
    if output_path is None:
        output_path = submission_path
    
    sub_df.to_csv(output_path, index=False)
    
    # 最终验证
    print(f"\n✓ 修复完成！保存到: {output_path}")
    print(f"\n最终检查:")
    print(f"  ✓ 行数: {len(sub_df)} (应与测试集相同: {len(test_df)})")
    print(f"  ✓ ID匹配: {set(test_df['id']) == set(sub_df['id'])}")
    print(f"  ✓ 列名: {list(sub_df.columns)}")
    print(f"  ✓ 无缺失值: {sub_df['trip_duration'].isna().sum() == 0}")
    print(f"  ✓ 无负值: {(sub_df['trip_duration'] < 0).sum() == 0}")
    print(f"  ✓ 预测值范围: [{sub_df['trip_duration'].min():.2f}, {sub_df['trip_duration'].max():.2f}] 秒")
    
    return sub_df


def main():
    parser = argparse.ArgumentParser(description='修复提交文件')
    parser.add_argument('--submission', type=str, required=True,
                       help='提交文件路径')
    parser.add_argument('--test', type=str, required=True,
                       help='测试集路径')
    parser.add_argument('--output', type=str,
                       help='输出文件路径（默认覆盖原文件）')
    
    args = parser.parse_args()
    
    submission_path = Path(args.submission)
    test_path = Path(args.test)
    output_path = Path(args.output) if args.output else None
    
    if not submission_path.exists():
        print(f"错误: 提交文件不存在: {submission_path}")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"错误: 测试文件不存在: {test_path}")
        sys.exit(1)
    
    fix_submission(submission_path, test_path, output_path)


if __name__ == "__main__":
    main()


