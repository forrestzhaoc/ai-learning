"""
数据处理和特征工程
包括数据清洗、特征转换、特征创建等
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


class DiabetesDataProcessor:
    """糖尿病数据处理器"""
    
    def __init__(self, handle_zeros='median', scaler_type='standard'):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        handle_zeros : str, default='median'
            处理零值的方法: 'median', 'mean', 'remove'
        scaler_type : str, default='standard'
            缩放方法: 'standard', 'robust', None
        """
        self.handle_zeros = handle_zeros
        self.scaler_type = scaler_type
        self.scaler = None
        self.zero_replace_values = {}
        
        # 这些特征不应该为0
        self.zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 
                                   'Insulin', 'BMI']
    
    def fit(self, df, target_col='Outcome'):
        """
        拟合数据处理器
        
        Parameters:
        -----------
        df : DataFrame
            训练数据
        target_col : str
            目标变量列名
        """
        # 计算零值替换值
        for col in self.zero_not_accepted:
            if col in df.columns:
                non_zero_values = df[df[col] != 0][col]
                if self.handle_zeros == 'median':
                    self.zero_replace_values[col] = non_zero_values.median()
                elif self.handle_zeros == 'mean':
                    self.zero_replace_values[col] = non_zero_values.mean()
        
        # 准备特征用于拟合scaler
        df_processed = df.copy()
        df_processed = self._handle_zeros(df_processed)
        df_processed = self._create_features(df_processed)
        
        # 获取特征列
        feature_cols = [col for col in df_processed.columns if col != target_col]
        X = df_processed[feature_cols]
        
        # 拟合scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
            self.scaler.fit(X)
        
        return self
    
    def transform(self, df, target_col='Outcome', is_test=False):
        """
        转换数据
        
        Parameters:
        -----------
        df : DataFrame
            要转换的数据
        target_col : str
            目标变量列名
        is_test : bool
            是否是测试集
        
        Returns:
        --------
        X : array
            特征矩阵
        y : array (如果不是测试集)
            目标变量
        feature_names : list
            特征名称列表
        """
        df_processed = df.copy()
        
        # 移除ID列（如果存在）
        if 'Id' in df_processed.columns:
            df_processed = df_processed.drop('Id', axis=1)
        
        # 处理零值
        df_processed = self._handle_zeros(df_processed)
        
        # 创建特征
        df_processed = self._create_features(df_processed)
        
        # 分离特征和目标
        if is_test or target_col not in df_processed.columns:
            feature_cols = df_processed.columns.tolist()
            X = df_processed[feature_cols]
            y = None
        else:
            feature_cols = [col for col in df_processed.columns if col != target_col]
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
            return X, y, feature_names
        else:
            return X, feature_names
    
    def fit_transform(self, df, target_col='Outcome'):
        """拟合并转换数据"""
        self.fit(df, target_col)
        return self.transform(df, target_col)
    
    def _handle_zeros(self, df):
        """处理零值"""
        df = df.copy()
        
        if self.handle_zeros == 'remove':
            # 移除包含零值的行
            for col in self.zero_not_accepted:
                if col in df.columns:
                    df = df[df[col] != 0]
        else:
            # 用统计值替换零值
            for col in self.zero_not_accepted:
                if col in df.columns and col in self.zero_replace_values:
                    df.loc[df[col] == 0, col] = self.zero_replace_values[col]
        
        return df
    
    def _create_features(self, df):
        """创建新特征"""
        df = df.copy()
        
        # BMI分类
        if 'BMI' in df.columns:
            df['BMI_Category'] = pd.cut(df['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100],
                                        labels=[0, 1, 2, 3])
            df['BMI_Category'] = df['BMI_Category'].astype(float)
        
        # 年龄分组
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'],
                                     bins=[0, 30, 40, 50, 100],
                                     labels=[0, 1, 2, 3])
            df['Age_Group'] = df['Age_Group'].astype(float)
        
        # BMI和年龄的交互特征
        if 'BMI' in df.columns and 'Age' in df.columns:
            df['BMI_Age'] = df['BMI'] * df['Age']
        
        # 血糖和胰岛素的比率
        if 'Glucose' in df.columns and 'Insulin' in df.columns:
            df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
        
        # 怀孕次数和年龄的比率
        if 'Pregnancies' in df.columns and 'Age' in df.columns:
            df['Pregnancies_Age_Ratio'] = df['Pregnancies'] / (df['Age'] + 1)
        
        # 血压和年龄
        if 'BloodPressure' in df.columns and 'Age' in df.columns:
            df['BP_Age'] = df['BloodPressure'] * df['Age']
        
        # 葡萄糖水平分类
        if 'Glucose' in df.columns:
            df['Glucose_Level'] = pd.cut(df['Glucose'],
                                         bins=[0, 100, 125, 200],
                                         labels=[0, 1, 2])
            df['Glucose_Level'] = df['Glucose_Level'].astype(float)
        
        return df


def load_and_process_data(train_path='data/train.csv',
                          test_path='data/test.csv',
                          handle_zeros='median',
                          scaler_type='standard'):
    """
    加载并处理训练和测试数据
    
    Returns:
    --------
    X_train, y_train, X_test, feature_names, processor
    """
    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 创建处理器
    processor = DiabetesDataProcessor(handle_zeros=handle_zeros,
                                      scaler_type=scaler_type)
    
    # 处理训练数据
    X_train, y_train, feature_names = processor.fit_transform(train_df)
    
    # 处理测试数据
    X_test, _ = processor.transform(test_df, is_test=True)
    
    print(f"\n处理后的训练集形状: {X_train.shape}")
    print(f"处理后的测试集形状: {X_test.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    return X_train, y_train, X_test, feature_names, processor


def split_train_val(X, y, test_size=0.2, random_state=42):
    """分割训练集和验证集"""
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, stratify=y)


if __name__ == '__main__':
    # 测试数据处理
    X_train, y_train, X_test, feature_names, processor = load_and_process_data()
    print(f"\n特征名称: {feature_names}")
    print(f"目标变量分布: {np.bincount(y_train.astype(int))}")

