import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = "processed_data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = np.load(f"{DATA_DIR}/X_train.npy")
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

print("Data loaded successfully")

print("\nTraining Logistic Regression...")

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

lr = LogisticRegression(max_iter=1000)

grid_lr = GridSearchCV(
    lr,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

joblib.dump(best_lr, f"{MODEL_DIR}/logistic_regression_v1.pkl")

print("Best Logistic Regression Params:", grid_lr.best_params_)

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

joblib.dump(rf, f"{MODEL_DIR}/random_forest_v1.pkl")

print("Random Forest trained successfully")
