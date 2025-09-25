
"""Load saved RandomForest model and run inference on sample test vectors."""
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score

def main():
    model_path = "artifacts/pm_model.joblib"
    if not os.path.exists(model_path):
        print("Model not found. Run train_pm.py first.")
        return
    clf = joblib.load(model_path)
    X = np.load("artifacts/test_X.npy")
    y = np.load("artifacts/test_y.npy")
    preds = clf.predict(X)
    print("Sample predictions:", preds)
    print("Ground truth:      ", y)
    print("Sample accuracy:", (preds==y).mean())

if __name__ == '__main__':
    main()
