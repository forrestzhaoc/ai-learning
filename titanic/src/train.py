"""
Titanic Kaggle solution - Focus on generalization.

Usage:
    python src/train.py --train data/train.csv --test data/test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class TitanicFeatureBuilder(BaseEstimator, TransformerMixin):
    """Minimal, robust feature engineering."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Extract title
        df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()
        df["Title"] = df["Title"].replace({
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
            "Col": "Rare", "Major": "Rare", "Dr": "Rare",
            "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
            "Don": "Rare", "Dona": "Rare"
        })
        df["Title"] = df["Title"].fillna("Rare")
        
        # Family size
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        
        # Woman or Child (strongest signal)
        df["IsWomanOrChild"] = ((df["Sex"] == "female") | (df["Title"] == "Master")).astype(int)
        
        # Simplified family categories
        df["FamilyCat"] = "Medium"
        df.loc[df["FamilySize"] == 1, "FamilyCat"] = "Alone"
        df.loc[df["FamilySize"] >= 5, "FamilyCat"] = "Large"
        
        # Fare per person
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        df["FarePerPerson"] = df["FarePerPerson"].replace([np.inf, -np.inf], np.nan)
        
        # Cabin
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].str[0].fillna("U")
        
        # Age categories
        df["AgeGroup"] = "Adult"
        df.loc[df["Age"] <= 16, "AgeGroup"] = "Child"
        df.loc[df["Age"] > 60, "AgeGroup"] = "Senior"
        df["AgeGroup"] = df["AgeGroup"].fillna("Adult")
        
        # Fare quartiles
        df["FareQuartile"] = pd.qcut(df["Fare"].fillna(df["Fare"].median()), q=4, labels=False, duplicates="drop")

        selected_cols = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "Title", "FamilySize", "IsAlone", "IsWomanOrChild", "FamilyCat",
            "FarePerPerson", "HasCabin", "Deck", "AgeGroup", "FareQuartile"
        ]
        return df[selected_cols]


def build_model(model_type: str, random_state: int) -> Pipeline:
    numeric_features = [
        "Age", "SibSp", "Parch", "Fare", "FamilySize", 
        "FarePerPerson", "HasCabin", "FareQuartile"
    ]
    categorical_features = [
        "Pclass", "Sex", "Embarked", "Title", "IsAlone", 
        "IsWomanOrChild", "FamilyCat", "Deck", "AgeGroup"
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    classifier = _build_classifier(model_type, random_state)

    return Pipeline([
        ("features", TitanicFeatureBuilder()),
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


def _build_classifier(model_type: str, random_state: int):
    
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    
    if model_type == "gb":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=random_state,
        )
    
    if model_type == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=6,
            reg_lambda=5.0,
            reg_alpha=2.0,
            gamma=2.0,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=random_state,
        )
    
    if model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            random_state=random_state,
        )
    
    if model_type == "voting":
        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                n_jobs=-1, random_state=random_state
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=3,
                min_samples_leaf=10, subsample=0.8, random_state=random_state
            )),
            ("lr", LogisticRegression(
                max_iter=1000, C=0.5, solver="lbfgs", random_state=random_state
            )),
        ]
        if XGBClassifier is not None:
            estimators.append(("xgb", XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                reg_lambda=5, gamma=2, n_jobs=-1, random_state=random_state
            )))
        
        return VotingClassifier(estimators, voting="soft", n_jobs=-1)
    
    # Default: simple voting
    return VotingClassifier([
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            n_jobs=-1, random_state=random_state
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=random_state
        )),
        ("lr", LogisticRegression(max_iter=1000, C=0.5, random_state=random_state)),
    ], voting="soft", n_jobs=-1)


def load_data(train_path: Path, test_path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    y = train_df[target_col]
    X = train_df.drop(columns=[target_col])
    test_df = pd.read_csv(test_path)
    return X, y, test_df


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Titanic model")
    parser.add_argument("--train", default="data/train.csv", type=Path)
    parser.add_argument("--test", default="data/test.csv", type=Path)
    parser.add_argument("--submission", default="submissions/titanic_submission.csv", type=Path)
    parser.add_argument("--model-path", default="models/titanic_model.joblib", type=Path)
    parser.add_argument("--target", default="Survived")
    parser.add_argument("--cv-folds", default=10, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--model-type", default="voting", choices=["rf", "gb", "xgb", "lr", "voting"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.model_type, args.random_state)
    X_train, y_train, X_test = load_data(args.train, args.test, args.target)

    print(f"Starting {args.cv_folds}-fold CV...")
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    print("Training final model...")
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Train accuracy: {train_acc:.4f}")

    ensure_parent(args.model_path)
    joblib.dump(model, args.model_path)
    print(f"Model saved to {args.model_path}")

    preds = model.predict(X_test)
    submission = pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": preds})

    ensure_parent(args.submission)
    submission.to_csv(args.submission, index=False)
    print(f"Submission saved to {args.submission}")


if __name__ == "__main__":
    main()
