# House Prices - Advanced Regression Techniques

This project implements a complete solution for the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It handles feature engineering, model training, cross-validation, and submission file generation.

## Project Layout

- `requirements.txt` – Python dependencies
- `src/train.py` – Main training script with CLI interface
- `src/data_processing.py` – Data preprocessing and feature engineering utilities
- `data/` – Expected location for `train.csv` and `test.csv` downloaded from Kaggle (create manually)
- `models/` – Trained model artifacts (created automatically)
- `submissions/` – Generated submission CSVs (created automatically)
- `predictions/` – Prediction results (created automatically)

## Setup

```bash
cd /home/ubuntu/projects/ai-learning/house-prices
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place Kaggle's `train.csv` and `test.csv` files under `data/`.

## Usage

### Basic Usage

```bash
python src/train.py
```

This will use the default XGBoost model with 5-fold cross-validation.

### Advanced Usage

```bash
python src/train.py \
  --train data/train.csv \
  --test data/test.csv \
  --model-type xgb \
  --cv-folds 5 \
  --random-state 42 \
  --model-path models/house_prices_model.joblib \
  --submission submissions/house_prices_submission.csv
```

### Model Types

- `xgb` (default) – XGBoost Regressor with early stopping
- `rf` – Random Forest Regressor
- `gbr` – Gradient Boosting Regressor
- `stacking` – Voting ensemble of XGBoost, Random Forest, and Gradient Boosting

### Example

```bash
# Use Random Forest
python src/train.py --model-type rf --cv-folds 10

# Use stacking ensemble
python src/train.py --model-type stacking
```

## Features

### Data Preprocessing

- Automatic missing value imputation (median for numeric, mode for categorical)
- Label encoding for categorical features
- Feature alignment between train and test sets

### Feature Engineering

- **TotalSF**: Total square footage (basement + 1st floor + 2nd floor)
- **TotalBathrooms**: Total number of bathrooms
- **TotalPorchSF**: Total porch square footage
- **HouseAge**: Age of the house (if year data available)
- **RemodAge**: Years since remodeling
- **HasBasement**: Binary indicator for basement
- **Has2ndFloor**: Binary indicator for second floor
- **HasGarage**: Binary indicator for garage
- **HasFireplace**: Binary indicator for fireplace
- **HasPool**: Binary indicator for pool

### Model Evaluation

- K-fold cross-validation with RMSE scoring (matching Kaggle's evaluation metric)
- Training set RMSE reporting
- Model persistence using joblib

## Notes

- The competition uses **RMSE** (Root Mean Squared Error) as the evaluation metric
- Predictions are clipped to be non-negative (house prices cannot be negative)
- The default XGBoost model uses early stopping to prevent overfitting
- Adjust hyperparameters in `build_model()` function to experiment with different configurations
- Add more feature engineering in `engineer_features()` to improve scores

## Competition Details

- **Competition**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Task**: Regression (predicting house sale prices)
- **Metric**: RMSE (Root Mean Squared Error)
- **Dataset**: Contains 79 explanatory variables describing various aspects of residential homes

