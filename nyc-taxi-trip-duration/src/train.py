"""
NYC Taxi Trip Duration 模型训练脚本

用法:
    python src/train.py --train data/train.csv --test data/test.csv
    python src/train.py --train data/train.csv --test data/test.csv --model-type xgb --cv-folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_processing import NYCFeatureBuilder, remove_outliers, clean_data


def load_data(train_path: Path, test_path: Path | None = None,
              sample_size: int | None = None):
    """加载数据"""
    print(f"加载训练数据: {train_path}")
    train_df = pd.read_csv(train_path, nrows=sample_size)
    print(f"✓ 训练数据: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
    
    test_df = None
    if test_path and test_path.exists():
        print(f"加载测试数据: {test_path}")
        test_df = pd.read_csv(test_path, nrows=sample_size)
        print(f"✓ 测试数据: {test_df.shape[0]} 行 × {test_df.shape[1]} 列")
    
    return train_df, test_df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None,
                 remove_outliers_flag: bool = True):
    """数据预处理和特征工程"""
    # 数据清洗
    print("\n数据清洗...")
    train_df = clean_data(train_df)
    
    # 移除异常值
    if remove_outliers_flag and 'trip_duration' in train_df.columns:
        print("\n移除异常值...")
        train_df = remove_outliers(train_df, target_col='trip_duration')
    
    # 分离特征和目标
    target_col = 'trip_duration'
    if target_col in train_df.columns:
        y_train = train_df[target_col]
        # 移除不应该用于训练的列
        cols_to_drop = [target_col, 'id', 'dropoff_datetime']
        X_train = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])
    else:
        raise ValueError(f"目标列 '{target_col}' 不存在")
    
    # 对测试集做同样的清洗
    X_test = None
    test_ids_all = None  # 保存所有原始ID
    if test_df is not None:
        # 先保存所有原始ID（在清洗之前）
        test_ids_all = test_df['id'].copy() if 'id' in test_df.columns else None
        
        # 数据清洗（可能会移除一些记录）
        test_df_cleaned = clean_data(test_df)
        # 保留清洗后的id列
        test_ids_cleaned = test_df_cleaned['id'].copy() if 'id' in test_df_cleaned.columns else None
        # 移除不应该用于训练的列
        cols_to_drop = ['id', 'dropoff_datetime']
        X_test = test_df_cleaned.drop(columns=[col for col in cols_to_drop if col in test_df_cleaned.columns])
        test_ids = test_ids_cleaned  # 用于预测的ID
    else:
        test_ids = None
    
    # 特征工程
    print("\n特征工程...")
    feature_builder = NYCFeatureBuilder(remove_outliers=False)
    feature_builder.fit(X_train, y_train)
    X_train = feature_builder.transform(X_train)
    
    if X_test is not None:
        X_test = feature_builder.transform(X_test)
    
    print(f"✓ 特征工程完成，特征数量: {X_train.shape[1]}")
    print(f"特征列表: {list(X_train.columns)[:10]}..." if len(X_train.columns) > 10 
          else f"特征列表: {list(X_train.columns)}")
    
    return X_train, y_train, X_test, test_ids, test_ids_all


def get_categorical_numeric_features(X: pd.DataFrame):
    """识别分类和数值特征"""
    categorical_features = []
    numeric_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    
    return categorical_features, numeric_features


def build_preprocessor(X: pd.DataFrame):
    """构建数据预处理管道"""
    categorical_features, numeric_features = get_categorical_numeric_features(X)
    
    preprocessor_steps = []
    
    # 数值特征处理
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preprocessor_steps.append(('num', numeric_transformer, numeric_features))
    
    # 分类特征处理
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor_steps.append(('cat', categorical_transformer, categorical_features))
    
    if not preprocessor_steps:
        return None
    
    preprocessor = ColumnTransformer(
        transformers=preprocessor_steps,
        remainder='passthrough'
    )
    
    return preprocessor


def build_model(model_type: str = 'xgb', random_state: int = 42, use_log_target: bool = False):
    """构建模型"""
    if model_type == 'xgb':
        if XGBRegressor is None:
            raise ImportError("xgboost 未安装。请运行: pip install xgboost")
        # 优化参数：更多树，更低的学习率，更好的正则化
        model = XGBRegressor(
            n_estimators=500 if use_log_target else 300,
            max_depth=8,
            learning_rate=0.05 if use_log_target else 0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            tree_method='hist'  # 更快，适合大数据
        )
    
    elif model_type == 'lgb':
        if LGBMRegressor is None:
            raise ImportError("lightgbm 未安装。请运行: pip install lightgbm")
        # LightGBM通常比XGBoost更快且效果好
        model = LGBMRegressor(
            n_estimators=500 if use_log_target else 300,
            max_depth=8,
            learning_rate=0.05 if use_log_target else 0.08,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state
        )
    
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=random_state)
    
    elif model_type == 'voting':
        models = []
        if XGBRegressor is not None:
            models.append(('xgb', XGBRegressor(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=random_state, n_jobs=-1, verbosity=0
            )))
        if LGBMRegressor is not None:
            models.append(('lgb', LGBMRegressor(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=random_state, n_jobs=-1, verbosity=-1
            )))
        models.append(('rf', RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=random_state, n_jobs=-1
        )))
        
        if not models:
            raise ValueError("需要至少安装 xgboost 或 lightgbm 来使用 voting 模型")
        
        model = VotingRegressor(estimators=models)
    
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return model


def rmsle(y_true, y_pred):
    """计算RMSLE（Root Mean Squared Logarithmic Error）"""
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def evaluate_model(y_true, y_pred, prefix=""):
    """评估模型性能"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # RMSLE（这是Kaggle评估指标）
    rmsle_score = rmsle(y_true, y_pred)
    
    # 使用对数变换后的MAE
    log_mae = mean_absolute_error(np.log1p(y_true), np.log1p(y_pred))
    
    print(f"\n{prefix}评估指标:")
    print(f"  MAE: {mae:.2f} 秒 ({mae/60:.2f} 分钟)")
    print(f"  RMSE: {rmse:.2f} 秒 ({rmse/60:.2f} 分钟)")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSLE: {rmsle_score:.5f} ⭐ (Kaggle评估指标)")
    print(f"  Log MAE: {log_mae:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'rmsle': rmsle_score,
        'log_mae': log_mae
    }


def train_model(X_train, y_train, model_type: str = 'xgb',
                cv_folds: int = 5, random_state: int = 42,
                use_preprocessor: bool = True, use_log_target: bool = False):
    """训练模型"""
    print(f"\n训练 {model_type.upper()} 模型...")
    if use_log_target:
        print("使用对数目标变量训练（优化RMSLE）...")
    
    # 转换目标变量为对数空间（用于RMSLE优化）
    if use_log_target:
        y_train_log = np.log1p(y_train)  # log(1+y)
    else:
        y_train_log = y_train
    
    # 构建预处理器
    if use_preprocessor:
        print("构建预处理管道...")
        preprocessor = build_preprocessor(X_train)
        
        if preprocessor is not None:
            # 拟合预处理器
            X_train_processed = preprocessor.fit_transform(X_train)
            print(f"✓ 预处理完成，特征数量: {X_train_processed.shape[1]}")
        else:
            X_train_processed = X_train.values
            preprocessor = None
    else:
        # 不使用sklearn预处理（树模型可以直接处理原始特征）
        X_train_processed = X_train
        preprocessor = None
    
    # 构建模型
    model = build_model(model_type=model_type, random_state=random_state, 
                       use_log_target=use_log_target)
    
    # 交叉验证
    if cv_folds > 1:
        print(f"\n执行 {cv_folds}-fold 交叉验证...")
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # 对于树模型，可以直接使用原始特征
        if model_type in ['xgb', 'lgb', 'rf', 'gbm', 'voting'] and use_preprocessor:
            X_train_for_cv = X_train
        else:
            X_train_for_cv = X_train_processed
        
        if isinstance(X_train_for_cv, pd.DataFrame):
            X_train_for_cv = X_train_for_cv.values
        
        cv_scores = cross_val_score(
            model, X_train_for_cv, y_train_log,
            cv=kf, scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        if use_log_target:
            print(f"✓ 交叉验证 Log MAE: {cv_mae:.4f} ± {cv_std:.4f}")
        else:
            print(f"✓ 交叉验证 MAE: {cv_mae:.2f} ± {cv_std:.2f} 秒")
    
    # 训练最终模型
    print("\n训练最终模型...")
    if model_type in ['xgb', 'lgb', 'rf', 'gbm', 'voting'] and use_preprocessor:
        # 树模型直接使用原始特征
        model.fit(X_train, y_train_log)
    else:
        model.fit(X_train_processed, y_train_log)
    
    # 训练集评估（预测后转换回原始空间）
    if model_type in ['xgb', 'lgb', 'rf', 'gbm', 'voting'] and use_preprocessor:
        y_pred_train_log = model.predict(X_train)
    else:
        y_pred_train_log = model.predict(X_train_processed)
    
    # 转换回原始空间
    if use_log_target:
        y_pred_train = np.expm1(y_pred_train_log)  # exp(y) - 1
    else:
        y_pred_train = y_pred_train_log
    
    evaluate_model(y_train, y_pred_train, "训练集")
    
    return model, preprocessor


def predict(model, X_test, preprocessor=None, model_type: str = 'xgb',
            use_preprocessor: bool = True, use_log_target: bool = False):
    """进行预测"""
    if model_type in ['xgb', 'lgb', 'rf', 'gbm', 'voting'] and use_preprocessor:
        # 树模型直接使用原始特征
        predictions_log = model.predict(X_test)
    else:
        if preprocessor is not None:
            X_test_processed = preprocessor.transform(X_test)
        else:
            X_test_processed = X_test.values
        predictions_log = model.predict(X_test_processed)
    
    # 转换回原始空间
    if use_log_target:
        predictions = np.expm1(predictions_log)  # exp(y) - 1
    else:
        predictions = predictions_log
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='NYC Taxi Trip Duration 模型训练')
    parser.add_argument('--train', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--test', type=str,
                       help='测试数据路径')
    parser.add_argument('--model-type', type=str, default='xgb',
                       choices=['xgb', 'lgb', 'rf', 'gbm', 'ridge', 'voting'],
                       help='模型类型')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--model-path', type=str,
                       help='模型保存路径')
    parser.add_argument('--submission', type=str,
                       help='提交文件路径')
    parser.add_argument('--sample-size', type=int,
                       help='采样数据大小（用于快速测试）')
    parser.add_argument('--remove-outliers', action='store_true', default=True,
                       help='是否移除异常值')
    parser.add_argument('--no-preprocessor', action='store_true', default=True,
                       help='不使用sklearn预处理器（仅用于树模型，默认True）')
    parser.add_argument('--use-log-target', action='store_true',
                       help='使用对数目标变量训练（针对RMSLE优化，推荐）')
    parser.add_argument('--random-state', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 路径处理
    train_path = Path(args.train)
    test_path = Path(args.test) if args.test else None
    
    if not train_path.exists():
        print(f"错误: 训练文件不存在: {train_path}")
        sys.exit(1)
    
    # 加载数据
    train_df, test_df = load_data(train_path, test_path, 
                                   sample_size=args.sample_size)
    
    # 数据预处理
    X_train, y_train, X_test, test_ids, test_ids_all = prepare_data(
        train_df, test_df, remove_outliers_flag=args.remove_outliers
    )
    
    # 训练模型
    model, preprocessor = train_model(
        X_train, y_train,
        model_type=args.model_type,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        use_preprocessor=not args.no_preprocessor,
        use_log_target=args.use_log_target
    )
    
    # 保存模型
    if args.model_path:
        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和预处理器
        model_dict = {
            'model': model,
            'preprocessor': preprocessor,
            'model_type': args.model_type,
            'use_preprocessor': not args.no_preprocessor,
            'use_log_target': args.use_log_target,
            'feature_names': list(X_train.columns)
        }
        joblib.dump(model_dict, model_path)
        print(f"\n✓ 模型已保存到: {model_path}")
    
    # 预测和生成提交文件
    if X_test is not None and args.submission:
        print("\n生成预测...")
        predictions = predict(
            model, X_test, preprocessor,
            model_type=args.model_type,
            use_preprocessor=not args.no_preprocessor,
            use_log_target=args.use_log_target
        )
        
        # 确保预测值为正数
        predictions = np.maximum(predictions, 1.0)
        
        # 创建提交文件
        submission_path = Path(args.submission)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备提交数据框（仅包含已预测的ID）
        if test_ids is not None:
            submission_df = pd.DataFrame({
                'id': test_ids,
                'trip_duration': predictions
            })
        else:
            submission_df = pd.DataFrame({
                'trip_duration': predictions
            })
        
        # 如果有原始测试集ID，确保所有ID都在提交文件中
        if test_ids_all is not None and len(test_ids_all) > len(test_ids):
            # 找出缺失的ID
            missing_ids = set(test_ids_all) - set(test_ids)
            if len(missing_ids) > 0:
                # 使用预测值的中位数填充缺失的ID
                median_pred = np.median(predictions)
                missing_df = pd.DataFrame({
                    'id': list(missing_ids),
                    'trip_duration': median_pred
                })
                submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
                print(f"  为 {len(missing_ids)} 个缺失ID填充预测值: {median_pred:.2f} 秒")
            
            # 按照原始测试集的顺序排序
            submission_df = submission_df.set_index('id').reindex(test_ids_all).reset_index()
        
        submission_df.to_csv(submission_path, index=False)
        print(f"✓ 提交文件已保存到: {submission_path}")
        print(f"  总行数: {len(submission_df)}")
        print(f"  预测范围: [{predictions.min():.2f}, {predictions.max():.2f}] 秒")


if __name__ == "__main__":
    main()
