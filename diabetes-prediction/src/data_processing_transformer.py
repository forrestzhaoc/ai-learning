"""
为Transformer模型准备数据
分离数值特征和分类特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class TabularDataset(Dataset):
    """表格数据Dataset"""
    
    def __init__(self, numeric_features, categorical_features, labels=None):
        self.numeric_features = torch.FloatTensor(numeric_features) if numeric_features is not None else None
        self.categorical_features = torch.LongTensor(categorical_features) if categorical_features is not None else None
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        if self.numeric_features is not None:
            return len(self.numeric_features)
        elif self.categorical_features is not None:
            return len(self.categorical_features)
        else:
            return 0
    
    def __getitem__(self, idx):
        item = {}
        
        if self.numeric_features is not None:
            item['numeric'] = self.numeric_features[idx]
        else:
            item['numeric'] = torch.zeros(1)
        
        if self.categorical_features is not None:
            item['categorical'] = self.categorical_features[idx]
        else:
            item['categorical'] = torch.zeros(1).long()
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
        
        return item


class TransformerDataProcessor:
    """为Transformer准备数据的处理器"""
    
    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = [
            'gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status'
        ]
        self.categorical_feature_names = []
        self.numeric_feature_names = []
        self.categorical_cardinalities = []
    
    def _create_features(self, df):
        """创建特征（复用数据处理器的逻辑）"""
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
    
    def fit(self, train_df, target_col='diagnosed_diabetes'):
        """拟合处理器"""
        print("拟合Transformer数据处理器...")
        
        train_processed = train_df.copy()
        
        # 编码类别变量
        for col in self.categorical_columns:
            if col in train_processed.columns:
                le = LabelEncoder()
                le.fit(train_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # 编码类别变量
        for col in self.categorical_columns:
            if col in train_processed.columns and col in self.label_encoders:
                train_processed[col] = self.label_encoders[col].transform(
                    train_processed[col].astype(str)
                )
        
        # 创建特征
        train_processed = self._create_features(train_processed)
        
        # 获取特征列
        feature_cols = [col for col in train_processed.columns 
                       if col not in [target_col, 'id']]
        
        # 定义分类特征（包括原始的类别特征和创建的分类特征）
        categorical_feature_names = [
            'gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status',
            'bmi_category', 'age_group', 'bp_category', 'sleep_quality',
            'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
        ]
        
        # 只保留实际存在的分类特征
        categorical_cols = [col for col in categorical_feature_names 
                          if col in feature_cols]
        
        # 数值特征是剩余的特征
        numeric_cols = [col for col in feature_cols 
                       if col not in categorical_cols]
        
        self.categorical_feature_names = categorical_cols
        self.numeric_feature_names = numeric_cols
        
        print(f"分类特征 ({len(categorical_cols)}): {categorical_cols[:5]}...")
        print(f"数值特征 ({len(numeric_cols)}): {numeric_cols[:5]}...")
        
        # 处理分类特征：计算每个特征的基数
        self.categorical_cardinalities = []
        for col in categorical_cols:
            if col in train_processed.columns:
                max_val = int(train_processed[col].max()) + 1
                min_val = int(train_processed[col].min())
                cardinality = max_val - min_val + 1
                # 确保至少为2
                cardinality = max(2, cardinality)
                self.categorical_cardinalities.append(cardinality)
        
        # 拟合数值特征scaler
        if len(numeric_cols) > 0:
            self.numeric_scaler.fit(train_processed[numeric_cols].fillna(0).values)
        
        print(f"分类特征基数: {self.categorical_cardinalities}")
        print(f"处理器拟合完成!")
        
        return self
    
    def transform(self, df, target_col='diagnosed_diabetes', is_test=False):
        """转换数据"""
        df_processed = df.copy()
        
        # 保存ID
        ids = df_processed['id'].values if 'id' in df_processed.columns else None
        
        # 编码类别变量
        for col in self.categorical_columns:
            if col in df_processed.columns and col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(
                    df_processed[col].astype(str)
                )
        
        # 创建特征
        df_processed = self._create_features(df_processed)
        
        # 处理分类特征
        categorical_data = None
        if len(self.categorical_feature_names) > 0:
            categorical_data = []
            for col in self.categorical_feature_names:
                if col in df_processed.columns:
                    # 确保值是连续的整数
                    values = df_processed[col].fillna(0).astype(int).values
                    values = np.clip(values, 0, None)  # 确保非负
                    categorical_data.append(values)
            
            if categorical_data:
                categorical_data = np.stack(categorical_data, axis=1)  # [N, num_cat_features]
        
        # 处理数值特征
        numeric_data = None
        if len(self.numeric_feature_names) > 0:
            numeric_values = df_processed[self.numeric_feature_names].fillna(0).values
            numeric_data = self.numeric_scaler.transform(numeric_values)
        
        # 获取目标变量
        if is_test or target_col not in df_processed.columns:
            labels = None
        else:
            labels = df_processed[target_col].values
        
        return numeric_data, categorical_data, labels, ids
    
    def fit_transform(self, df, target_col='diagnosed_diabetes'):
        """拟合并转换数据"""
        self.fit(df, target_col)
        return self.transform(df, target_col)


def create_data_loaders(
    train_df,
    val_df,
    processor,
    batch_size=256,
    target_col='diagnosed_diabetes'
):
    """创建数据加载器"""
    # 处理训练数据
    train_numeric, train_categorical, train_labels, _ = processor.fit_transform(train_df, target_col)
    
    # 处理验证数据
    val_numeric, val_categorical, val_labels, _ = processor.transform(val_df, target_col)
    
    # 创建数据集
    train_dataset = TabularDataset(train_numeric, train_categorical, train_labels)
    val_dataset = TabularDataset(val_numeric, val_categorical, val_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, processor







