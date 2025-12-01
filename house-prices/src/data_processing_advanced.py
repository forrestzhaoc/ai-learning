"""
House Prices 高级数据处理和特征工程模块

改进点：
1. 更细致的缺失值处理
2. 异常值检测和处理
3. 更丰富的特征工程（多项式特征、交互特征等）
4. 特征选择和降维
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, RobustScaler


def handle_missing_values(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """更细致的缺失值处理"""
    df = df.copy()
    
    # 对于某些特征，NA 本身有意义（表示"没有"）
    for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 对于地下室和车库相关的数值特征，NA 表示 0
    for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 对于年份特征，用 0 或特殊值填充
    for col in ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # LotFrontage 用邻里的中位数填充
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # MSZoning 用众数填充
    if 'MSZoning' in df.columns:
        df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    
    # Utilities 几乎都是 AllPub，直接删除
    if 'Utilities' in df.columns:
        df = df.drop(columns=['Utilities'])
    
    # Functional 用 Typ 填充
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')
    
    # Exterior 用众数填充
    for col in ['Exterior1st', 'Exterior2nd']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Electrical 用众数填充
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # KitchenQual 用众数填充
    if 'KitchenQual' in df.columns:
        df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    
    # SaleType 用众数填充
    if 'SaleType' in df.columns:
        df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    
    # MasVnrType 和 MasVnrArea
    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    return df


def remove_outliers(df: pd.DataFrame, target: pd.Series = None) -> tuple[pd.DataFrame, pd.Series]:
    """移除异常值（仅用于训练集）"""
    if target is None:
        return df, None
    
    # 移除 GrLivArea 和 SalePrice 的明显异常值
    outlier_idx = df[(df['GrLivArea'] > 4000) & (target < 300000)].index
    
    df = df.drop(outlier_idx)
    target = target.drop(outlier_idx)
    
    return df, target


def engineer_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """高级特征工程"""
    df = df.copy()
    
    # === 基础特征 ===
    # 总面积
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # 总浴室数
    df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                            df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    # 总门廊面积
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                          df['3SsnPorch'] + df['ScreenPorch'])
    
    # 房屋年龄
    if 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        df['GarageAge'] = df['GarageAge'].replace([np.inf, -np.inf], 0)
    
    # === 二元特征 ===
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int)
    df['HasWoodDeck'] = (df['WoodDeckSF'] > 0).astype(int)
    df['HasBsmtFinished'] = (df['BsmtFinSF1'] > 0).astype(int)
    df['HasMasVnr'] = (df['MasVnrArea'] > 0).astype(int)
    
    # === 质量和条件评分 ===
    # 总体质量评分
    qual_cols = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 
                 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
                 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    # 将质量特征转换为数值
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in qual_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].map(quality_map).fillna(0)
    
    # 总质量评分
    df['TotalQual'] = df['OverallQual'] + df['OverallCond']
    if 'ExterQual' in df.columns:
        df['TotalQual'] += df['ExterQual']
    if 'KitchenQual' in df.columns:
        df['TotalQual'] += df['KitchenQual']
    
    # === 比例特征 ===
    # 地下室占比
    df['BsmtRatio'] = df['TotalBsmtSF'] / (df['TotalSF'] + 1)
    
    # 生活面积占比
    df['LivingAreaRatio'] = df['GrLivArea'] / (df['TotalSF'] + 1)
    
    # 车库占地面积比
    df['GarageRatio'] = df['GarageArea'] / (df['LotArea'] + 1)
    
    # === 交互特征 ===
    df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['OverallQual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['OverallQual_TotalBathrooms'] = df['OverallQual'] * df['TotalBathrooms']
    df['GrLivArea_TotalBathrooms'] = df['GrLivArea'] * df['TotalBathrooms']
    
    # === 邻里特征 ===
    # 简化邻里分类
    if 'Neighborhood' in df.columns:
        good_neighborhoods = ['NoRidge', 'NridgHt', 'StoneBr']
        df['IsGoodNeighborhood'] = df['Neighborhood'].isin(good_neighborhoods).astype(int)
    
    # === 多项式特征（对重要数值特征）===
    for col in ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageArea']:
        if col in df.columns:
            df[f'{col}_Squared'] = df[col] ** 2
            df[f'{col}_Cubed'] = df[col] ** 3
            df[f'{col}_Sqrt'] = np.sqrt(df[col])
    
    return df


def encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """编码特征"""
    # 合并以保持一致的编码
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 获取分类特征
    categorical_cols = combined.select_dtypes(include=['object']).columns
    
    # Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        label_encoders[col] = le
    
    # 分离训练集和测试集
    train_encoded = combined.iloc[:n_train].copy()
    test_encoded = combined.iloc[n_train:].copy()
    
    return train_encoded, test_encoded


def handle_skewness(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    threshold: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """处理数值特征的偏态分布"""
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    
    # 计算偏度
    skewed_features = []
    for col in numeric_cols:
        if col in train_df.columns:
            # 只对正值特征计算偏度
            col_data = train_df[col].dropna()
            if (col_data >= 0).all() and col_data.std() > 0:
                skewness = skew(col_data)
                if abs(skewness) > threshold:
                    skewed_features.append(col)
    
    # 对偏态特征进行 log1p 转换
    for col in skewed_features:
        if col in train_df.columns:
            # 确保值为非负
            train_df[col] = train_df[col].clip(lower=0)
            train_df[col] = np.log1p(train_df[col])
        if col in test_df.columns:
            test_df[col] = test_df[col].clip(lower=0)
            test_df[col] = np.log1p(test_df[col])
    
    print(f"对 {len(skewed_features)} 个偏态特征进行了 log1p 转换")
    
    return train_df, test_df


def prepare_data_advanced(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'SalePrice',
    remove_outliers_flag: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """完整的高级数据预处理流程"""
    # 提取目标变量
    y_train = train_df[target_col].copy() if target_col in train_df.columns else None
    
    # 删除不需要的列
    drop_cols = ['Id', target_col] if target_col in train_df.columns else ['Id']
    train_df = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
    test_df = test_df.drop(columns=['Id'] if 'Id' in test_df.columns else [])
    
    # 处理缺失值
    print("处理缺失值...")
    train_df = handle_missing_values(train_df, is_train=True)
    test_df = handle_missing_values(test_df, is_train=False)
    
    # 移除异常值（仅训练集）
    if remove_outliers_flag and y_train is not None:
        print("移除异常值...")
        train_df, y_train = remove_outliers(train_df, y_train)
        train_df = train_df.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
    
    # 特征工程
    print("特征工程...")
    train_df = engineer_features_advanced(train_df)
    test_df = engineer_features_advanced(test_df)
    
    # 编码分类特征
    print("编码分类特征...")
    train_df, test_df = encode_features(train_df, test_df)
    
    # 处理偏态分布
    print("处理偏态分布...")
    train_df, test_df = handle_skewness(train_df, test_df, threshold=0.75)
    
    # 确保列对齐
    common_cols = train_df.columns.intersection(test_df.columns)
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]
    
    # 填充任何剩余的 NaN
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # 替换无穷值
    train_df = train_df.replace([np.inf, -np.inf], 0)
    test_df = test_df.replace([np.inf, -np.inf], 0)
    
    print(f"最终特征数量: {len(common_cols)}")
    
    return train_df, test_df, y_train

