"""
大规模糖尿病数据集处理
处理包含70万训练样本和30万测试样本的数据集
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class LargeDiabetesDataProcessor:
    """大规模糖尿病数据处理器"""
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoders = {}
        self.categorical_columns = [
            'gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status'
        ]
    
    def fit(self, df, target_col='diagnosed_diabetes'):
        """拟合数据处理器"""
        print("拟合数据处理器...")
        
        df_processed = df.copy()
        
        # 编码类别变量
        for col in self.categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                le.fit(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # 处理并创建特征
        df_processed = self._encode_categorical(df_processed)
        df_processed = self._create_features(df_processed)
        
        # 获取特征列
        feature_cols = [col for col in df_processed.columns 
                       if col not in [target_col, 'id']]
        X = df_processed[feature_cols]
        
        # 拟合scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        
        print(f"处理器拟合完成，特征数量: {len(feature_cols)}")
        return self
    
    def transform(self, df, target_col='diagnosed_diabetes', is_test=False):
        """转换数据"""
        df_processed = df.copy()
        
        # 保存ID
        ids = df_processed['id'].values if 'id' in df_processed.columns else None
        
        # 编码类别变量
        df_processed = self._encode_categorical(df_processed)
        
        # 创建特征
        df_processed = self._create_features(df_processed)
        
        # 分离特征和目标
        feature_cols = [col for col in df_processed.columns 
                       if col not in [target_col, 'id']]
        
        if is_test or target_col not in df_processed.columns:
            X = df_processed[feature_cols]
            y = None
        else:
            X = df_processed[feature_cols]
            y = df_processed[target_col].values
        
        # 获取特征名称
        feature_names = X.columns.tolist()
        
        # 缩放
        if self.scaler is not None:
            X = self.scaler.transform(X)
        else:
            X = X.values
        
        if y is not None:
            return X, y, feature_names, ids
        else:
            return X, feature_names, ids
    
    def fit_transform(self, df, target_col='diagnosed_diabetes'):
        """拟合并转换数据"""
        self.fit(df, target_col)
        return self.transform(df, target_col)
    
    def _encode_categorical(self, df):
        """编码类别变量"""
        df = df.copy()
        
        for col in self.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _create_features(self, df):
        """创建新特征"""
        df = df.copy()
        
        # BMI分类
        if 'bmi' in df.columns:
            df['bmi_category'] = pd.cut(df['bmi'], 
                                        bins=[0, 18.5, 25, 30, 100],
                                        labels=[0, 1, 2, 3]).astype(float)
        
        # 年龄分组
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                     bins=[0, 30, 40, 50, 60, 100],
                                     labels=[0, 1, 2, 3, 4]).astype(float)
        
        # 血压类别
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            df['bp_category'] = ((df['systolic_bp'] >= 140) | 
                                (df['diastolic_bp'] >= 90)).astype(int)
            df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
        
        # 胆固醇比率
        if 'hdl_cholesterol' in df.columns and 'ldl_cholesterol' in df.columns:
            df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
        
        # 生活方式评分
        if 'physical_activity_minutes_per_week' in df.columns and 'diet_score' in df.columns:
            df['lifestyle_score'] = (df['physical_activity_minutes_per_week'] / 150) * df['diet_score']
        
        # 睡眠质量指标
        if 'sleep_hours_per_day' in df.columns:
            df['sleep_quality'] = ((df['sleep_hours_per_day'] >= 7) & 
                                  (df['sleep_hours_per_day'] <= 9)).astype(int)
        
        # 代谢健康指标
        if 'bmi' in df.columns and 'waist_to_hip_ratio' in df.columns:
            df['metabolic_risk'] = df['bmi'] * df['waist_to_hip_ratio']
        
        # 心血管风险
        if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
            df['cardiovascular_risk'] = (df['heart_rate'] * df['systolic_bp']) / 10000
        
        # 风险因素计数
        risk_cols = ['family_history_diabetes', 'hypertension_history', 
                    'cardiovascular_history']
        if all(col in df.columns for col in risk_cols):
            df['risk_factors_count'] = df[risk_cols].sum(axis=1)
        
        # 年龄与BMI的交互
        if 'age' in df.columns and 'bmi' in df.columns:
            df['age_bmi_interaction'] = df['age'] * df['bmi']
        
        return df


def load_and_process_data(train_path='data/train.csv',
                          test_path='data/test.csv',
                          sample_size=None):
    """
    加载并处理大规模数据
    
    Parameters:
    -----------
    sample_size : int, optional
        如果指定，则只使用部分数据进行快速测试
    """
    print("=" * 70)
    print("加载大规模糖尿病数据集")
    print("=" * 70)
    
    # 加载数据
    print("\n加载训练数据...")
    if sample_size:
        train_df = pd.read_csv(train_path, nrows=sample_size)
        print(f"使用采样数据: {sample_size}条")
    else:
        train_df = pd.read_csv(train_path)
    
    print(f"训练集形状: {train_df.shape}")
    
    print("\n加载测试数据...")
    test_df = pd.read_csv(test_path)
    print(f"测试集形状: {test_df.shape}")
    
    # 显示数据信息
    print(f"\n特征列: {len(train_df.columns) - 2} 个")  # 减去id和target
    print(f"目标变量分布:")
    print(train_df['diagnosed_diabetes'].value_counts())
    print(f"\n目标变量比例:")
    print(train_df['diagnosed_diabetes'].value_counts(normalize=True))
    
    # 创建处理器
    print("\n创建数据处理器...")
    processor = LargeDiabetesDataProcessor(scaler_type='standard')
    
    # 处理训练数据
    print("\n处理训练数据...")
    X_train, y_train, feature_names, train_ids = processor.fit_transform(train_df)
    
    # 处理测试数据
    print("处理测试数据...")
    X_test, test_feature_names, test_ids = processor.transform(test_df, is_test=True)
    
    print(f"\n处理后的训练集形状: {X_train.shape}")
    print(f"处理后的测试集形状: {X_test.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    return X_train, y_train, X_test, feature_names, processor, test_ids


if __name__ == '__main__':
    # 测试数据处理（使用小样本）
    X_train, y_train, X_test, feature_names, processor, test_ids = \
        load_and_process_data(sample_size=10000)
    
    print(f"\n测试完成!")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"特征: {len(feature_names)}")

