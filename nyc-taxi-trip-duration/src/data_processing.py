"""
NYC Taxi Trip Duration 数据预处理和特征工程模块
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class NYCFeatureBuilder(BaseEstimator, TransformerMixin):
    """NYC Taxi Trip Duration 特征工程器"""
    
    def __init__(self, remove_outliers=True, outlier_threshold=0.99):
        """
        参数:
            remove_outliers: 是否移除异常值
            outlier_threshold: 异常值阈值（百分位数）
        """
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.vendor_encoder_ = None
        self.vendor_classes_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """训练阶段：计算统计量和编码器"""
        if y is not None:
            # 计算目标变量的异常值阈值
            self.duration_threshold_ = y.quantile(self.outlier_threshold)
            self.duration_min_ = y.quantile(0.01)  # 移除极端小值
        
        # 拟合vendor_id编码器
        if 'vendor_id' in X.columns:
            self.vendor_encoder_ = LabelEncoder()
            self.vendor_encoder_.fit(X['vendor_id'].astype(str))
            self.vendor_classes_ = set(self.vendor_encoder_.classes_)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换阶段：特征工程"""
        df = X.copy()
        
        # 日期时间特征
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            
            # 基础时间特征
            df['hour'] = df['pickup_datetime'].dt.hour
            df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
            df['day_of_month'] = df['pickup_datetime'].dt.day
            df['month'] = df['pickup_datetime'].dt.month
            df['year'] = df['pickup_datetime'].dt.year
            df['quarter'] = df['pickup_datetime'].dt.quarter
            
            # 是否为周末
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # 是否为月初/月末
            df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
            df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
            
            # 时间段分类
            df['time_of_day'] = pd.cut(
                df['hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['night', 'morning', 'afternoon', 'evening'],
                include_lowest=True
            ).astype(str)
            
            # 是否为高峰时段（早高峰和晚高峰）
            df['is_rush_hour'] = (
                ((df['hour'] >= 7) & (df['hour'] <= 9)) |
                ((df['hour'] >= 17) & (df['hour'] <= 19))
            ).astype(int)
            
            # 是否为深夜/凌晨
            df['is_late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            
            # 循环编码（帮助模型理解时间的周期性）
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 地理距离特征
        if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude',
                                              'dropoff_latitude', 'dropoff_longitude']):
            # 坐标差值
            df['lat_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
            df['lon_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
            
            # 曼哈顿距离
            df['manhattan_distance'] = (
                np.abs(df['lat_diff']) + np.abs(df['lon_diff'])
            )
            
            # Haversine距离（球面距离，单位：公里）
            df['haversine_distance'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'],
                df['dropoff_latitude'], df['dropoff_longitude']
            )
            
            # 欧几里得距离（直线距离）
            df['euclidean_distance'] = np.sqrt(
                df['lat_diff']**2 + df['lon_diff']**2
            )
            
            # 方向角度
            df['direction'] = np.arctan2(
                df['lat_diff'],
                df['lon_diff']
            )
            # 方向的正弦和余弦（循环编码）
            df['direction_sin'] = np.sin(df['direction'])
            df['direction_cos'] = np.cos(df['direction'])
            
            # 到中心点的距离（NYC大致中心：40.7580, -73.9855）
            nyc_center_lat, nyc_center_lon = 40.7580, -73.9855
            df['pickup_distance_to_center'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'],
                nyc_center_lat, nyc_center_lon
            )
            df['dropoff_distance_to_center'] = self._haversine_distance(
                df['dropoff_latitude'], df['dropoff_longitude'],
                nyc_center_lat, nyc_center_lon
            )
            
            # 到主要机场的距离（JFK, LaGuardia, Newark）
            jfk_lat, jfk_lon = 40.6413, -73.7781
            lga_lat, lga_lon = 40.7769, -73.8740
            ewr_lat, ewr_lon = 40.6895, -74.1745
            
            df['pickup_distance_to_jfk'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'], jfk_lat, jfk_lon
            )
            df['pickup_distance_to_lga'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'], lga_lat, lga_lon
            )
            df['pickup_distance_to_ewr'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'], ewr_lat, ewr_lon
            )
            df['dropoff_distance_to_jfk'] = self._haversine_distance(
                df['dropoff_latitude'], df['dropoff_longitude'], jfk_lat, jfk_lon
            )
            df['dropoff_distance_to_lga'] = self._haversine_distance(
                df['dropoff_latitude'], df['dropoff_longitude'], lga_lat, lga_lon
            )
            df['dropoff_distance_to_ewr'] = self._haversine_distance(
                df['dropoff_latitude'], df['dropoff_longitude'], ewr_lat, ewr_lon
            )
            
            # 到曼哈顿下城的距离（金融区）
            downtown_lat, downtown_lon = 40.7074, -74.0113
            df['pickup_distance_to_downtown'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'],
                downtown_lat, downtown_lon
            )
            df['dropoff_distance_to_downtown'] = self._haversine_distance(
                df['dropoff_latitude'], df['dropoff_longitude'],
                downtown_lat, downtown_lon
            )
            
            # 是否为JFK/LaGuardia相关行程（机场行程通常时间较长）
            df['pickup_near_airport'] = (
                (df['pickup_distance_to_jfk'] < 5) |
                (df['pickup_distance_to_lga'] < 5) |
                (df['pickup_distance_to_ewr'] < 5)
            ).astype(int)
            df['dropoff_near_airport'] = (
                (df['dropoff_distance_to_jfk'] < 5) |
                (df['dropoff_distance_to_lga'] < 5) |
                (df['dropoff_distance_to_ewr'] < 5)
            ).astype(int)
        
        # 乘客数量特征
        if 'passenger_count' in df.columns:
            df['passenger_count'] = df['passenger_count'].fillna(1)
            df['passenger_count'] = df['passenger_count'].clip(0, 9)  # 限制范围
            df['has_passengers'] = (df['passenger_count'] > 0).astype(int)
        
        # Vendor特征（转换为数值）
        if 'vendor_id' in df.columns:
            # 使用fit阶段保存的编码器或创建新的
            if self.vendor_encoder_ is not None:
                # 处理未见过的类别
                vendor_str = df['vendor_id'].astype(str)
                unknown_mask = ~vendor_str.isin(self.vendor_classes_)
                if unknown_mask.any():
                    # 对于未知类别，使用最常见的类别编码
                    vendor_str[unknown_mask] = self.vendor_encoder_.classes_[0]
                df['vendor_id'] = self.vendor_encoder_.transform(vendor_str)
            else:
                # 如果没有fit过，直接尝试转换为int（通常vendor_id是数字）
                try:
                    df['vendor_id'] = df['vendor_id'].astype(int)
                except:
                    # 如果转换失败，使用标签编码
                    le = LabelEncoder()
                    df['vendor_id'] = le.fit_transform(df['vendor_id'].astype(str))
                    self.vendor_encoder_ = le
                    self.vendor_classes_ = set(le.classes_)
        
        # Store and forward flag（已经是数值，保持不变）
        if 'store_and_fwd_flag' in df.columns:
            df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({
                'Y': 1, 'N': 0, 'y': 1, 'n': 0
            }).fillna(0).astype(int)
        
        # Time of day转换为数值编码
        if 'time_of_day' in df.columns:
            time_mapping = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
            df['time_of_day'] = df['time_of_day'].map(time_mapping).fillna(0).astype(int)
        
        # 交互特征（特征组合）
        if 'haversine_distance' in df.columns:
            # 距离与时间的交互
            if 'hour' in df.columns:
                df['distance_hour'] = df['haversine_distance'] * df['hour']
                df['distance_is_weekend'] = df['haversine_distance'] * df['is_weekend']
                df['distance_is_rush_hour'] = df['haversine_distance'] * df['is_rush_hour']
            
            # 距离与乘客数的交互
            if 'passenger_count' in df.columns:
                df['distance_passengers'] = df['haversine_distance'] * df['passenger_count']
            
            # 距离与vendor的交互
            if 'vendor_id' in df.columns:
                df['distance_vendor'] = df['haversine_distance'] * df['vendor_id']
        
        # 速度估算特征（基于距离，虽然我们不知道时间，但可以用平均速度估算）
        if 'haversine_distance' in df.columns:
            # 假设平均速度30-50 km/h，用于特征
            df['estimated_duration_based_on_distance'] = df['haversine_distance'] * 120  # 约2分钟/公里
        
        # 移除原始日期时间列（保留特征）
        if 'pickup_datetime' in df.columns:
            df = df.drop(columns=['pickup_datetime'])
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """
        计算Haversine距离（两点之间的球面距离）
        返回距离（单位：公里）
        """
        R = 6371  # 地球半径（公里）
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) *
             np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c


def remove_outliers(df: pd.DataFrame, target_col: str = 'trip_duration',
                    lower_percentile: float = 0.01,
                    upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    移除异常值
    
    参数:
        df: 数据框
        target_col: 目标列名
        lower_percentile: 下界百分位数
        upper_percentile: 上界百分位数
    
    返回:
        移除异常值后的数据框
    """
    if target_col not in df.columns:
        return df
    
    lower_bound = df[target_col].quantile(lower_percentile)
    upper_bound = df[target_col].quantile(upper_percentile)
    
    mask = (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)
    removed_count = (~mask).sum()
    
    if removed_count > 0:
        print(f"移除了 {removed_count} 个异常值 "
              f"({removed_count/len(df)*100:.2f}%)")
        print(f"范围: [{lower_bound:.2f}, {upper_bound:.2f}] 秒")
    
    return df[mask].reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗
    
    参数:
        df: 原始数据框
    
    返回:
        清洗后的数据框
    """
    df = df.copy()
    
    # 移除明显无效的坐标（NYC大致范围）
    nyc_lat_min, nyc_lat_max = 40.5, 41.0
    nyc_lon_min, nyc_lon_max = -74.3, -73.7
    
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude',
                                          'dropoff_latitude', 'dropoff_longitude']):
        mask = (
            (df['pickup_latitude'].between(nyc_lat_min, nyc_lat_max)) &
            (df['pickup_longitude'].between(nyc_lon_min, nyc_lon_max)) &
            (df['dropoff_latitude'].between(nyc_lat_min, nyc_lat_max)) &
            (df['dropoff_longitude'].between(nyc_lon_min, nyc_lon_max))
        )
        
        removed = (~mask).sum()
        if removed > 0:
            print(f"移除了 {removed} 条无效坐标记录 "
                  f"({removed/len(df)*100:.2f}%)")
            df = df[mask].reset_index(drop=True)
    
    # 移除零距离的行程
    if all(col in df.columns for col in ['pickup_latitude', 'pickup_longitude',
                                          'dropoff_latitude', 'dropoff_longitude']):
        lat_diff = np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
        lon_diff = np.abs(df['dropoff_longitude'] - df['pickup_longitude'])
        zero_distance = (lat_diff < 1e-6) & (lon_diff < 1e-6)
        
        removed = zero_distance.sum()
        if removed > 0:
            print(f"移除了 {removed} 条零距离行程记录")
            df = df[~zero_distance].reset_index(drop=True)
    
    return df
