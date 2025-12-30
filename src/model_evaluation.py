import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

DATA_DIR = "processed_data"
MODEL_DIR = "models"

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

models = {
    "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic_regression_v1.pkl"),
    "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest_v1.pkl")
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

df_results = pd.DataFrame(results)
print("\nModel Comparison:")
print(df_results)

from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

print("\nPerforming Cross-Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_model = joblib.load("models/logistic_regression_v1.pkl")

cv_scores = cross_val_score(
    lr_model,
    np.load("processed_data/X_train.npy"),
    np.load("processed_data/y_train.npy"),
    cv=skf,
    scoring="f1"
)

print("Cross-Validation F1 Scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

