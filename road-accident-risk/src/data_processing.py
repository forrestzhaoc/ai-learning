"""
Data Processing and Feature Engineering for Road Accident Risk Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataProcessor:
    """Process and prepare data for modeling"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load training and test data"""
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        
        print(f"Loading data from {self.data_dir}...")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def extract_datetime_features(self, df, datetime_col='DateTime'):
        """Extract features from datetime column"""
        if datetime_col not in df.columns:
            return df
        
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['year'] = df[datetime_col].dt.year
        
        # Create time-based features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
                               ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
        
        # Drop original datetime column
        df = df.drop(columns=[datetime_col])
        
        return df
    
    def handle_missing_values(self, df, is_train=True):
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if is_train:
                    self.fill_values_num = self.fill_values_num or {}
                    self.fill_values_num[col] = df[col].median()
                df[col].fillna(self.fill_values_num.get(col, df[col].median()), inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if is_train:
                    self.fill_values_cat = self.fill_values_cat or {}
                    self.fill_values_cat[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(self.fill_values_cat.get(col, 'Unknown'), inplace=True)
        
        return df
    
    def encode_categorical_features(self, df, is_train=True):
        """Encode categorical features"""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df[col] = -1
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Check if required columns exist before creating interactions
        if 'hour' in df.columns and 'Weather_Condition' in df.columns:
            df['hour_weather'] = df['hour'] * df['Weather_Condition']
        
        if 'day_of_week' in df.columns and 'Road_Type' in df.columns:
            df['day_road'] = df['day_of_week'] * df['Road_Type']
        
        if 'is_rush_hour' in df.columns and 'Road_Type' in df.columns:
            df['rush_road'] = df['is_rush_hour'] * df['Road_Type']
        
        return df
    
    def prepare_features(self, train_df, test_df, target_col='accident_risk'):
        """Prepare features for modeling"""
        print("\n=== Feature Engineering ===")
        
        # Initialize fill values dictionaries
        self.fill_values_num = {}
        self.fill_values_cat = {}
        
        # Store ids
        test_ids = test_df['id'] if 'id' in test_df.columns else None
        
        # Separate target from training data
        if target_col in train_df.columns:
            y_train = train_df[target_col].values
            train_df = train_df.drop(columns=[target_col])
        else:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        # Remove id columns
        id_cols = ['id']
        train_df = train_df.drop(columns=[col for col in id_cols if col in train_df.columns])
        test_df = test_df.drop(columns=[col for col in id_cols if col in test_df.columns])
        
        # Extract datetime features
        print("Extracting datetime features...")
        train_df = self.extract_datetime_features(train_df)
        test_df = self.extract_datetime_features(test_df)
        
        # Handle missing values
        print("Handling missing values...")
        train_df = self.handle_missing_values(train_df, is_train=True)
        test_df = self.handle_missing_values(test_df, is_train=False)
        
        # Encode categorical features
        print("Encoding categorical features...")
        train_df = self.encode_categorical_features(train_df, is_train=True)
        test_df = self.encode_categorical_features(test_df, is_train=False)
        
        # Create interaction features
        print("Creating interaction features...")
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # Ensure train and test have same columns
        missing_in_test = set(train_df.columns) - set(test_df.columns)
        missing_in_train = set(test_df.columns) - set(train_df.columns)
        
        for col in missing_in_test:
            test_df[col] = 0
        
        for col in missing_in_train:
            train_df[col] = 0
        
        # Align column order
        test_df = test_df[train_df.columns]
        
        X_train = train_df.values
        X_test = test_df.values
        self.feature_names = train_df.columns.tolist()
        
        print(f"\nFinal feature shape: {X_train.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, test_ids
    
    def get_train_val_split(self, X_train, y_train, val_size=0.2, random_state=42):
        """Split training data into train and validation sets"""
        return train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=None  # Set to y_train if classification
        )

def load_and_process_data(data_dir='data', val_size=0.2):
    """
    Convenience function to load and process data in one step
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, test_ids, processor
    """
    processor = DataProcessor(data_dir=data_dir)
    
    # Load data
    train_df, test_df = processor.load_data()
    
    # Prepare features
    X_train_full, X_test, y_train_full, test_ids = processor.prepare_features(train_df, test_df)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = processor.get_train_val_split(
        X_train_full, y_train_full, val_size=val_size
    )
    
    print(f"\n=== Data Split ===")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, test_ids, processor

if __name__ == '__main__':
    # Test data processing
    X_train, X_val, X_test, y_train, y_val, test_ids, processor = load_and_process_data()
    print("\n✓ Data processing completed successfully!")
