#!/usr/bin/env python3
"""
Titanic Kaggle solution - 逻辑回归版本

用法:
    python src/train_logistic.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_simple import build_datasets


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SUBMISSION_DIR = ROOT_DIR / "submissions"


def _load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def _build_pipeline() -> Pipeline:
    # StandardScaler improves convergence for logistic regression on mixed-scale data.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def main() -> None:
    print("读取 Titanic 训练/测试数据...")
    train_df, test_df = _load_raw_data()

    y_train = train_df["Survived"]
    X_train, X_test = build_datasets(train_df, test_df)

    feature_cols = X_train.columns.tolist()

    pipeline = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("使用 5 折交叉验证评估逻辑回归...")
    cv_scores = cross_val_score(pipeline, X_train[feature_cols], y_train, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("训练最终模型...")
    pipeline.fit(X_train[feature_cols], y_train)
    train_acc = pipeline.score(X_train[feature_cols], y_train)
    print(f"训练集准确率: {train_acc:.4f}")

    print("对测试集进行预测并导出提交文件...")
    predictions = pipeline.predict(X_test[feature_cols])
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions,
    })

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "titanic_submission_logreg.csv"
    submission.to_csv(output_path, index=False)
    print(f"提交文件已保存: {output_path}")


if __name__ == "__main__":
    main()


