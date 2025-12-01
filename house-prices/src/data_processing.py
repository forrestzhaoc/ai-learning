"""
House Prices 数据处理和特征工程模块
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fill_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """填充缺失值"""
    # 数值型特征用中位数填充
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in train_df.columns:
            if train_df[col].isnull().any() or (col in test_df.columns and test_df[col].isnull().any()):
                median_val = train_df[col].median()
                # 如果中位数也是 NaN，用 0 填充
                if pd.isna(median_val):
                    median_val = 0
                train_df[col].fillna(median_val, inplace=True)
                if col in test_df.columns:
                    test_df[col].fillna(median_val, inplace=True)
    
    # 分类型特征用众数填充
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in train_df.columns:
            if train_df[col].isnull().any() or (col in test_df.columns and test_df[col].isnull().any()):
                mode_vals = train_df[col].mode()
                mode_val = mode_vals[0] if not mode_vals.empty else 'None'
                train_df[col].fillna(mode_val, inplace=True)
                if col in test_df.columns:
                    test_df[col].fillna(mode_val, inplace=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程"""
    out = df.copy()
    
    # 总居住面积
    out['TotalSF'] = out.get('TotalBsmtSF', 0) + out.get('1stFlrSF', 0) + out.get('2ndFlrSF', 0)
    
    # 总浴室数
    out['TotalBathrooms'] = (
        out.get('FullBath', 0) + 
        0.5 * out.get('HalfBath', 0) + 
        out.get('BsmtFullBath', 0) + 
        0.5 * out.get('BsmtHalfBath', 0)
    )
    
    # 总门廊面积
    out['TotalPorchSF'] = (
        out.get('OpenPorchSF', 0) + 
        out.get('EnclosedPorch', 0) + 
        out.get('3SsnPorch', 0) + 
        out.get('ScreenPorch', 0)
    )
    
    # 房屋年龄（如果存在销售年份）
    if 'YrSold' in out.columns and 'YearBuilt' in out.columns:
        out['HouseAge'] = out['YrSold'] - out['YearBuilt']
        out['RemodAge'] = out['YrSold'] - out.get('YearRemodAdd', out['YearBuilt'])
    
    # 是否有地下室
    if 'TotalBsmtSF' in out.columns:
        out['HasBasement'] = (out['TotalBsmtSF'] > 0).astype(int)
    
    # 是否有2楼
    if '2ndFlrSF' in out.columns:
        out['Has2ndFloor'] = (out['2ndFlrSF'] > 0).astype(int)
    
    # 是否有车库
    if 'GarageArea' in out.columns:
        out['HasGarage'] = (out['GarageArea'] > 0).astype(int)
    
    # 是否有壁炉
    if 'Fireplaces' in out.columns:
        out['HasFireplace'] = (out['Fireplaces'] > 0).astype(int)
    
    # 是否有游泳池
    if 'PoolArea' in out.columns:
        out['HasPool'] = (out['PoolArea'] > 0).astype(int)
    
    return out


def encode_categorical_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    label_encode: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """编码分类特征"""
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    categorical_cols = combined.select_dtypes(include=['object']).columns
    
    if label_encode:
        # 使用 LabelEncoder
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
            label_encoders[col] = le
    else:
        # 使用 one-hot encoding
        combined = pd.get_dummies(combined, columns=categorical_cols, dummy_na=True)
    
    # 分离训练集和测试集
    n_train = len(train_df)
    train_encoded = combined.iloc[:n_train].reset_index(drop=True)
    test_encoded = combined.iloc[n_train:].reset_index(drop=True)
    
    return train_encoded, test_encoded


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'SalePrice',
    drop_id_col: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """完整的数据预处理流程"""
    # 保存 ID 列（如果有）
    train_id = train_df.get('Id', None)
    test_id = test_df.get('Id', None)
    
    # 提取目标变量
    if target_col in train_df.columns:
        y_train = train_df[target_col].copy()
        train_df = train_df.drop(columns=[target_col])
    else:
        y_train = None
    
    # 删除 ID 列（如果存在且需要）
    if drop_id_col:
        if 'Id' in train_df.columns:
            train_df = train_df.drop(columns=['Id'])
        if 'Id' in test_df.columns:
            test_df = test_df.drop(columns=['Id'])
    
    # 填充缺失值
    fill_missing_values(train_df, test_df)
    
    # 特征工程
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # 编码分类特征
    train_encoded, test_encoded = encode_categorical_features(train_df, test_df, label_encode=True)
    
    # 确保训练集和测试集列对齐
    common_cols = train_encoded.columns.intersection(test_encoded.columns)
    train_encoded = train_encoded[common_cols]
    test_encoded = test_encoded[common_cols]
    
    return train_encoded, test_encoded, y_train

