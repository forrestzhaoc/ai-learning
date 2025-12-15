"""
Exploratory Data Analysis for Road Accident Risk Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_data(data_dir='data'):
    """Load the dataset"""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def basic_info(df, name='Dataset'):
    """Display basic information about the dataset"""
    print(f"\n{'='*60}")
    print(f"{name} Basic Information")
    print('='*60)
    
    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    print("\n" + "-"*60)
    print("Data Types:")
    print("-"*60)
    print(df.dtypes.value_counts())
    
    print("\n" + "-"*60)
    print("Missing Values:")
    print("-"*60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values!")
    
    print("\n" + "-"*60)
    print("Column Names:")
    print("-"*60)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    return missing_df

def analyze_target(df, target_col='accident_risk'):
    """Analyze the target variable"""
    print(f"\n{'='*60}")
    print(f"Target Variable Analysis: {target_col}")
    print('='*60)
    
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found in dataset")
        return
    
    print(f"\nTarget Statistics:")
    print(df[target_col].describe())
    
    print(f"\nTarget Distribution:")
    print(df[target_col].value_counts().sort_index())
    
    # Visualize target distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df[target_col].hist(bins=50, edgecolor='black')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.title(f'{target_col} Distribution')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    df[target_col].plot(kind='box', vert=False)
    plt.xlabel(target_col)
    plt.title(f'{target_col} Box Plot')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/target_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved target distribution plot to data/target_distribution.png")
    plt.close()

def analyze_categorical_features(df, target_col='accident_risk', max_categories=20):
    """Analyze categorical features"""
    print(f"\n{'='*60}")
    print("Categorical Features Analysis")
    print('='*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    print(f"\nFound {len(categorical_cols)} categorical features:")
    
    for col in categorical_cols:
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Top 5 values:")
        print(value_counts.head().to_string().replace('\n', '\n    '))
        
        if target_col in df.columns and df[col].nunique() <= max_categories:
            print(f"\n  Average {target_col} by {col}:")
            avg_by_cat = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            print(avg_by_cat.head().to_string().replace('\n', '\n    '))

def analyze_numerical_features(df, target_col='accident_risk'):
    """Analyze numerical features"""
    print(f"\n{'='*60}")
    print("Numerical Features Analysis")
    print('='*60)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    
    print(f"\nFound {len(numerical_cols)} numerical features")
    
    if len(numerical_cols) > 0:
        print("\nNumerical Features Statistics:")
        print(df[numerical_cols].describe())
    
    # Correlation with target
    if target_col in df.columns and len(numerical_cols) > 0:
        print(f"\nCorrelation with {target_col}:")
        correlations = df[numerical_cols + [target_col]].corr()[target_col].drop(target_col)
        correlations = correlations.abs().sort_values(ascending=False)
        print(correlations.to_string())
        
        # Visualize correlations
        plt.figure(figsize=(10, max(6, len(correlations) * 0.3)))
        correlations.sort_values().plot(kind='barh')
        plt.xlabel('Absolute Correlation with Severity')
        plt.title('Feature Correlations with Target')
        plt.tight_layout()
        plt.savefig('data/feature_correlations.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved correlation plot to data/feature_correlations.png")
        plt.close()

def create_datetime_analysis(df):
    """Analyze datetime patterns if datetime column exists"""
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not datetime_cols:
        return
    
    print(f"\n{'='*60}")
    print("DateTime Features Analysis")
    print('='*60)
    
    for col in datetime_cols:
        print(f"\n{col}:")
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"  Date range: {df[col].min()} to {df[col].max()}")
            
            # Extract and analyze time components
            df['hour'] = df[col].dt.hour
            df['day_of_week'] = df[col].dt.dayofweek
            df['month'] = df[col].dt.month
            
            if 'Severity' in df.columns:
                print(f"\n  Average Severity by Hour:")
                print(df.groupby('hour')['Severity'].mean().to_string().replace('\n', '\n    '))
                
                print(f"\n  Average Severity by Day of Week:")
                print(df.groupby('day_of_week')['Severity'].mean().to_string().replace('\n', '\n    '))
        except Exception as e:
            print(f"  Error processing datetime: {e}")

def generate_eda_report(data_dir='data'):
    """Generate complete EDA report"""
    print("\n" + "="*60)
    print("ROAD ACCIDENT RISK - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    train_df, test_df = load_data(data_dir)
    
    # Basic info
    basic_info(train_df, 'Training Data')
    basic_info(test_df, 'Test Data')
    
    # Analyze target variable
    analyze_target(train_df)
    
    # Analyze categorical features
    analyze_categorical_features(train_df)
    
    # Analyze numerical features
    analyze_numerical_features(train_df)
    
    # Analyze datetime patterns
    create_datetime_analysis(train_df.copy())
    
    print("\n" + "="*60)
    print("EDA Complete!")
    print("="*60)

if __name__ == '__main__':
    generate_eda_report()
