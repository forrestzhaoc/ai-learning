"""
快速开始脚本
一键运行完整的训练和提交流程
"""

import os
import sys

def main():
    """运行完整流程"""
    print("=" * 70)
    print("糖尿病预测 - 快速开始")
    print("=" * 70)
    
    # 检查数据文件
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        print("\n未找到数据文件。是否创建示例数据？")
        print("注意：如果您有真实的Kaggle数据，请将其放入data/目录后再运行。")
        choice = input("创建示例数据？(y/n, 默认y): ").strip().lower()
        
        if choice != 'n':
            print("\n" + "=" * 70)
            print("步骤 1: 创建示例数据")
            print("=" * 70)
            import create_sample_data
            create_sample_data.create_sample_data()
        else:
            print("\n请将数据文件放入data/目录:")
            print("  - data/train.csv")
            print("  - data/test.csv")
            return
    
    # 训练模型
    print("\n" + "=" * 70)
    print("步骤 2: 训练模型")
    print("=" * 70)
    print("\n开始训练模型（这可能需要几分钟）...\n")
    
    sys.path.append('src')
    from src.train import main as train_main
    train_main()
    
    # 生成提交文件
    print("\n" + "=" * 70)
    print("步骤 3: 生成提交文件")
    print("=" * 70)
    
    from generate_submission import generate_submission_ensemble
    generate_submission_ensemble()
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print("\n您的提交文件已生成在 submissions/ 目录")
    print("推荐使用: submissions/diabetes_submission_ensemble.csv")
    print("\n如需查看所有提交文件，可运行:")
    print("  python generate_submission.py")


if __name__ == '__main__':
    main()

