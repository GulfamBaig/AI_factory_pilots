
"""Train a simple classifier (RandomForest) on synthetic predictive maintenance data.
Outputs a model file and prints test accuracy.
"""
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from data_generator import generate

def main():
    os.makedirs("artifacts", exist_ok=True)
    df = generate(n_samples=7000, n_features=8, fault_ratio=0.12, random_state=1)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=3)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Test Accuracy:", acc)
    print(classification_report(y_test, preds))
    joblib.dump(clf, "artifacts/pm_model.joblib")
    # Save a small sample of test data for inference demo
    import numpy as np
    np.save("artifacts/test_X.npy", X_test[:20])
    np.save("artifacts/test_y.npy", y_test[:20])
    print("Model and sample test data saved in artifacts/")

if __name__ == '__main__':
    main()
