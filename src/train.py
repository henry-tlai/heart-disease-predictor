"""Reproduce the model comparison from the notebooks and run a feature ablation.

Usage (from the repo root):
    python src/train.py

Writes results/metrics.csv and prints a summary table.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "smart_healthcare_dataset.csv"
OUT = ROOT / "results" / "metrics.csv"

FEATURES = ["age", "gender", "bmi", "exercise_level", "smoking", "alcohol",
            "blood_pressure", "cholesterol", "glucose", "fatigue",
            "chest_pain", "dizziness", "diabetes", "stroke"]

# health_risk_score is left out everywhere (it is constant at 100).
FEATURE_SETS = {
    "all features": FEATURES,
    "no chest_pain/diabetes/stroke": [f for f in FEATURES if f not in ("chest_pain", "diabetes", "stroke")],
    "age only": ["age"],
}


def load():
    df = pd.read_csv(DATA)
    df["gender"] = (df["gender"] == "Male").astype(int)
    return df


def evaluate(df, features):
    X, y = df[features], df["heart_disease"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaler is fit on the training set only to avoid leakage.
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42), False),
        "SVM (RBF)": (SVC(class_weight="balanced", probability=True, random_state=42), False),
        "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=42), False),
    }

    rows = []
    for name, (model, scaled) in models.items():
        a, b = (X_tr_s, X_te_s) if scaled else (X_tr, X_te)
        model.fit(a, y_tr)
        pred, prob = model.predict(b), model.predict_proba(b)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, prob),
        })
    return rows


def main():
    df = load()
    results = []
    for label, feats in FEATURE_SETS.items():
        for row in evaluate(df, feats):
            results.append({"feature_set": label, **row})

    out = pd.DataFrame(results).round(4)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()