# Kaggle Titanic Baseline

This project implements a full pipeline for the classic [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).  
It handles feature engineering, model training, cross-validation, and submission file generation.

## Project Layout

- `requirements.txt` – Python dependencies.
- `src/train.py` – CLI entry point that trains and evaluates the model, then writes a Kaggle-ready submission.
- `src/train_simple.py` – Simplified training script with basic feature engineering.
- `src/train_transformer.py` – Transformer-based model (TabTransformer architecture).
- `src/eda.py` – Exploratory Data Analysis script for correlation analysis.
- `data/` – Expected location for `train.csv` and `test.csv` downloaded from Kaggle (create manually).
- `models/` – Trained model artifacts (created automatically).
- `submissions/` – Generated submission CSVs (created automatically).

## Setup

```bash
cd /home/ubuntu/projects/ai-learning/titanic
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place Kaggle's `train.csv` and `test.csv` files under `data/`.

## Usage

```bash
python src/train.py \
  --train data/train.csv \
  --test data/test.csv \
  --submission submissions/titanic_submission.csv \
  --model-path models/titanic_model.joblib \
  --model-type stacking \
  --cv-folds 5 \
  --random-state 7
```

Key flags:

- `--model-type` &mdash; choose among `stacking` (default, best accuracy), `xgb`, `voting`, or `hgb`.
- `--cv-folds` &mdash; Stratified K-folds used both for reporting accuracy and within the stacking meta-learner.

The script prints cross-validation accuracy, fits on the full training set, saves the estimator, and writes a submission file.

## Exploratory Data Analysis (EDA)

Run correlation analysis to understand feature relationships:

```bash
# Basic analysis
python src/eda.py

# Generate report file
python src/eda.py --output correlation_report.txt

# Generate visualizations (requires matplotlib and seaborn)
python src/eda.py --visualize

# Customize analysis
python src/eda.py --top-n 20 --threshold 0.6 --output report.txt
```

The EDA script provides:
- **Feature-target correlations**: Identifies which features are most correlated with survival
- **Feature-feature correlations**: Detects multicollinearity issues
- **Categorical analysis**: Analyzes survival rates by category
- **Visualizations**: Heatmaps, bar charts, and scatter plots (with `--visualize`)

## Transformer Model

Use a Transformer-based model (TabTransformer) for tabular data:

```bash
# Basic usage
python src/train_transformer.py

# Customize training
python src/train_transformer.py --epochs 50 --batch-size 32 --lr 0.001

# With cross-validation
python src/train_transformer.py --cv-folds 5 --epochs 30
```

**Key Features:**
- **TabTransformer Architecture**: Uses Transformer encoder to learn feature interactions
- **Categorical Embeddings**: Categorical features are embedded before Transformer processing
- **Numeric Projection**: Numeric features are projected to embedding space
- **Self-Attention**: Learns complex feature interactions automatically
- **Cross-Validation**: Built-in stratified K-fold cross-validation

**Architecture:**
1. Categorical features → Embedding layers → Transformer Encoder
2. Numeric features → Linear projection → Transformer Encoder
3. All features concatenated → MLP classifier → Binary output

**Note**: Requires PyTorch (`torch>=2.0`). Install with:
```bash
pip install torch
```

## Notes

- Adjust hyperparameters inside `build_model()` to experiment with different algorithms.
- The default preprocessing uses scikit-learn pipelines to encode categorical data and scale numeric features.
- Add more feature engineering or models (e.g., stacking) to improve leaderboard scores.

