# Kaggle Titanic Baseline

This project implements a full pipeline for the classic [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).  
It handles feature engineering, model training, cross-validation, and submission file generation.

## Project Layout

- `requirements.txt` – Python dependencies.
- `src/train.py` – CLI entry point that trains and evaluates the model, then writes a Kaggle-ready submission.
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

## Notes

- Adjust hyperparameters inside `build_model()` to experiment with different algorithms.
- The default preprocessing uses scikit-learn pipelines to encode categorical data and scale numeric features.
- Add more feature engineering or models (e.g., stacking) to improve leaderboard scores.

