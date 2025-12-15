"""
Training script with ensemble models for Road Accident Risk Prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pickle
import os
from datetime import datetime

class EnsembleModel:
    """Ensemble model combining XGBoost, LightGBM, and CatBoost"""
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {
            'xgboost': [],
            'lightgbm': [],
            'catboost': []
        }
        self.weights = None
        self.oof_predictions = None
        self.cv_scores = {}
        
    def get_xgboost_params(self):
        """Get XGBoost parameters"""
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'verbose': 0
        }
    
    def get_lightgbm_params(self):
        """Get LightGBM parameters"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 7,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'verbose': -1
        }
    
    def get_catboost_params(self):
        """Get CatBoost parameters"""
        return {
            'loss_function': 'RMSE',
            'learning_rate': 0.05,
            'depth': 7,
            'l2_leaf_reg': 3,
            'subsample': 0.8,
            'random_strength': 1,
            'bagging_temperature': 0.2,
            'random_state': self.random_state,
            'iterations': 1000,
            'early_stopping_rounds': 50,
            'verbose': 0
        }
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        params = self.get_xgboost_params()
        early_stopping_rounds = params.pop('early_stopping_rounds')
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        params = self.get_lightgbm_params()
        early_stopping_rounds = params.pop('early_stopping_rounds')
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        params = self.get_catboost_params()
        early_stopping_rounds = params.pop('early_stopping_rounds')
        
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        return model
    
    def train_fold(self, X_train, y_train, X_val, y_val, fold):
        """Train all models for a single fold"""
        print(f"\n  Fold {fold + 1}/{self.n_folds}")
        
        # Train XGBoost
        print("    Training XGBoost...", end=" ")
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        xgb_pred = xgb_model.predict(X_val)
        xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
        print(f"RMSE: {xgb_rmse:.4f}")
        self.models['xgboost'].append(xgb_model)
        
        # Train LightGBM
        print("    Training LightGBM...", end=" ")
        lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_pred = lgb_model.predict(X_val)
        lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
        print(f"RMSE: {lgb_rmse:.4f}")
        self.models['lightgbm'].append(lgb_model)
        
        # Train CatBoost
        print("    Training CatBoost...", end=" ")
        cb_model = self.train_catboost(X_train, y_train, X_val, y_val)
        cb_pred = cb_model.predict(X_val)
        cb_rmse = np.sqrt(mean_squared_error(y_val, cb_pred))
        print(f"RMSE: {cb_rmse:.4f}")
        self.models['catboost'].append(cb_model)
        
        return {
            'xgboost': xgb_pred,
            'lightgbm': lgb_pred,
            'catboost': cb_pred
        }
    
    def fit(self, X, y):
        """Train ensemble model with cross-validation"""
        print(f"\n{'='*60}")
        print("Training Ensemble Model")
        print(f"{'='*60}")
        print(f"Training samples: {len(X):,}")
        print(f"K-Fold CV: {self.n_folds} folds")
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        oof_predictions = {
            'xgboost': np.zeros(len(X)),
            'lightgbm': np.zeros(len(X)),
            'catboost': np.zeros(len(X))
        }
        
        fold_scores = {
            'xgboost': [],
            'lightgbm': [],
            'catboost': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train models for this fold
            predictions = self.train_fold(X_train, y_train, X_val, y_val, fold)
            
            # Store out-of-fold predictions
            for model_name, pred in predictions.items():
                oof_predictions[model_name][val_idx] = pred
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                fold_scores[model_name].append(rmse)
        
        # Calculate overall CV scores
        print(f"\n{'='*60}")
        print("Cross-Validation Results")
        print(f"{'='*60}")
        
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions[model_name]))
            avg_fold_rmse = np.mean(fold_scores[model_name])
            std_fold_rmse = np.std(fold_scores[model_name])
            
            self.cv_scores[model_name] = {
                'oof_rmse': oof_rmse,
                'avg_fold_rmse': avg_fold_rmse,
                'std_fold_rmse': std_fold_rmse
            }
            
            print(f"\n{model_name.upper()}:")
            print(f"  OOF RMSE: {oof_rmse:.4f}")
            print(f"  Avg Fold RMSE: {avg_fold_rmse:.4f} ± {std_fold_rmse:.4f}")
        
        # Optimize ensemble weights
        print(f"\n{'='*60}")
        print("Optimizing Ensemble Weights")
        print(f"{'='*60}")
        
        self.weights = self.optimize_weights(y, oof_predictions)
        
        # Calculate ensemble OOF score
        ensemble_pred = self.weighted_average(oof_predictions)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        
        print(f"\nEnsemble OOF RMSE: {ensemble_rmse:.4f}")
        
        self.oof_predictions = oof_predictions
        
        return self
    
    def optimize_weights(self, y_true, oof_predictions):
        """Optimize ensemble weights using simple grid search"""
        best_rmse = float('inf')
        best_weights = None
        
        # Try different weight combinations
        for w1 in np.arange(0.2, 0.5, 0.05):
            for w2 in np.arange(0.2, 0.5, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < 0.2 or w3 > 0.5:
                    continue
                
                weights = {
                    'xgboost': w1,
                    'lightgbm': w2,
                    'catboost': w3
                }
                
                pred = self.weighted_average(oof_predictions, weights)
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights
        
        print(f"Optimal weights: {best_weights}")
        print(f"Best RMSE: {best_rmse:.4f}")
        
        return best_weights
    
    def weighted_average(self, predictions, weights=None):
        """Calculate weighted average of predictions"""
        if weights is None:
            weights = self.weights
        
        if weights is None:
            # Equal weights
            weights = {
                'xgboost': 1/3,
                'lightgbm': 1/3,
                'catboost': 1/3
            }
        
        return (
            predictions['xgboost'] * weights['xgboost'] +
            predictions['lightgbm'] * weights['lightgbm'] +
            predictions['catboost'] * weights['catboost']
        )
    
    def predict(self, X):
        """Make predictions on new data"""
        predictions = {
            'xgboost': np.zeros(len(X)),
            'lightgbm': np.zeros(len(X)),
            'catboost': np.zeros(len(X))
        }
        
        # Average predictions from all folds
        for model_name in ['xgboost', 'lightgbm', 'catboost']:
            fold_preds = []
            for model in self.models[model_name]:
                fold_preds.append(model.predict(X))
            predictions[model_name] = np.mean(fold_preds, axis=0)
        
        # Return weighted ensemble prediction
        return self.weighted_average(predictions)
    
    def save(self, filepath):
        """Save the ensemble model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✓ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a saved ensemble model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return model

def train_model(X_train, y_train, n_folds=5, random_state=42, save_path='models/ensemble_model.pkl'):
    """
    Train the ensemble model
    
    Args:
        X_train: Training features
        y_train: Training target
        n_folds: Number of folds for cross-validation
        random_state: Random seed
        save_path: Path to save the trained model
    
    Returns:
        Trained ensemble model
    """
    # Create and train model
    model = EnsembleModel(n_folds=n_folds, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Save model
    model.save(save_path)
    
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set"""
    print(f"\n{'='*60}")
    print("Validation Set Evaluation")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return rmse, mae

if __name__ == '__main__':
    from data_processing import load_and_process_data
    
    # Load and process data
    X_train, X_val, X_test, y_train, y_val, test_ids, processor = load_and_process_data()
    
    # Train model using all training data (train + val)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Train ensemble model
    model = train_model(X_train_full, y_train_full, n_folds=5)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
