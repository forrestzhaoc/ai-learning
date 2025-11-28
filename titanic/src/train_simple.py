"""
Titanic Kaggle solution - 简洁版本（覆盖指定特征工程要求）

用法:
    python src/train_simple.py
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


MANDATORY_COLS: Sequence[str] = (
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Name",
    "Cabin",
    "Ticket",
)

AGE_BINS = [-1, 12, 18, 35, 55, 120]
AGE_LABELS = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]


def extract_title(series: pd.Series) -> pd.Series:
    titles = series.str.extract(r",\s*([^.]+)\.\s*", expand=False).str.strip()
    return titles.replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Dr": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
    }).fillna("Rare")


def _fill_missing(train_df: pd.DataFrame, other_df: pd.DataFrame) -> None:
    age_median = train_df["Age"].median()
    fare_median = train_df["Fare"].median()
    embarked_mode = train_df["Embarked"].mode(dropna=True).iloc[0]

    for df in (train_df, other_df):
        df["Age"] = df["Age"].fillna(age_median)
        df["Fare"] = df["Fare"].fillna(fare_median)
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
        df["Cabin"] = df["Cabin"].fillna("Unknown")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Title"] = extract_title(out["Name"])

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)

    out["FarePerPerson"] = (out["Fare"] / out["FamilySize"]).replace([np.inf, -np.inf], np.nan)
    out["FarePerPerson"] = out["FarePerPerson"].fillna(out["Fare"])
    fare_bins = pd.qcut(out["Fare"], q=4, labels=False, duplicates="drop")
    out["FareBin"] = fare_bins.astype(float).fillna(-1).astype(int)

    out["Deck"] = out["Cabin"].str[0].fillna("U")
    out["HasCabin"] = (out["Cabin"] != "Unknown").astype(int)
    out["CabinCount"] = out["Cabin"].str.split().str.len().fillna(0).astype(int)

    out["TicketPrefix"] = (
        out["Ticket"].str.replace(r"[0-9./ ]", "", regex=True).str.upper().replace("", "NONE")
    )
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("size")

    out = out.drop(columns=["Name", "Cabin", "Ticket"])
    return out


def encode_and_align(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = (
        pd.concat(
            [train_df.assign(_dataset="train"), test_df.assign(_dataset="test")],
            ignore_index=True,
        )
        .reset_index(drop=True)
    )

    # Embarked one-hot
    embarked_dummies = pd.get_dummies(combined["Embarked"], prefix="Embarked")
    combined = pd.concat([combined.drop(columns=["Embarked"]), embarked_dummies], axis=1)

    # Encode categorical columns
    label_cols = ["Title", "Deck", "TicketPrefix", "AgeBin"]
    for col in label_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    # Sex -> binary
    combined["Sex"] = combined["Sex"].map({"male": 0, "female": 1}).astype(int)

    # Ensure FarePerPerson with fill
    combined["FarePerPerson"] = combined["FarePerPerson"].fillna(combined["Fare"])

    train_processed = combined[combined["_dataset"] == "train"].drop(columns=["_dataset"])
    test_processed = combined[combined["_dataset"] == "test"].drop(columns=["_dataset"])
    return train_processed, test_processed


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["IsAlone_Sex"] = out["IsAlone"] * out["Sex"]
    out["IsAlone_Pclass"] = out["IsAlone"] * out["Pclass"]
    out["IsAlone_Age"] = out["IsAlone"] * out["Age"]

    out["Pclass_Sex"] = out["Pclass"] * out["Sex"]
    out["Pclass_Fare"] = out["Pclass"] * out["Fare"]
    out["Pclass_Age"] = out["Pclass"] * out["Age"]
    out["Pclass_IsAlone"] = out["Pclass"] * out["IsAlone"]

    out["Title_Sex"] = out["Title"] * out["Sex"]
    out["Title_Pclass"] = out["Title"] * out["Pclass"]
    out["Title_IsAlone"] = out["Title"] * out["IsAlone"]
    out["Title_Fare"] = out["Title"] * out["Fare"]

    out["AgeBin_Pclass"] = out["AgeBin"] * out["Pclass"]
    out["FareBin_Pclass"] = out["FareBin"] * out["Pclass"]
    return out


def build_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train_df[list(MANDATORY_COLS)].copy()
    test_features = test_df[list(MANDATORY_COLS)].copy()

    _fill_missing(train_features, test_features)

    train_features = engineer_features(train_features)
    test_features = engineer_features(test_features)

    train_features, test_features = encode_and_align(train_features, test_features)

    train_features = add_interactions(train_features)
    test_features = add_interactions(test_features)

    return train_features, test_features


def main() -> None:
    print("读取数据...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    y_train = train_df["Survived"]
    X_train, X_test = build_datasets(train_df, test_df)

    feature_cols = X_train.columns.tolist()

    print("训练模型并执行交叉验证...")
    base_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_model, X_train[feature_cols], y_train, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X_train[feature_cols], y_train)

    train_acc = model.score(X_train[feature_cols], y_train)
    print(f"训练准确率: {train_acc:.4f}")

    print("生成预测...")
    predictions = model.predict(X_test[feature_cols])

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions,
    })
    submission.to_csv("submissions/titanic_submission_simple.csv", index=False)
    print("提交文件已保存: submissions/titanic_submission_simple.csv")


if __name__ == "__main__":
    main()
