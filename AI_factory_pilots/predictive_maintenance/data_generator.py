
"""Synthetic sensor dataset generator for predictive maintenance.
Generates time-series-like tabular data with labeled 'healthy' vs 'fault' states.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate(n_samples=5000, n_features=6, fault_ratio=0.1, random_state=42):
    rng = np.random.RandomState(random_state)
    # healthy: gaussian around 0, small variance
    healthy_n = int(n_samples * (1 - fault_ratio))
    fault_n = n_samples - healthy_n
    healthy = rng.normal(loc=0.0, scale=1.0, size=(healthy_n, n_features))
    # faults: shift mean and add higher variance and sporadic spikes
    fault = rng.normal(loc=3.0, scale=1.5, size=(fault_n, n_features))
    # add occasional spikes
    spikes = (rng.rand(fault_n, n_features) < 0.02).astype(float) * rng.normal(10,5,(fault_n,n_features))
    fault += spikes
    X = np.vstack([healthy, fault])
    y = np.hstack([np.zeros(healthy_n,dtype=int), np.ones(fault_n,dtype=int)])
    # Shuffle
    idx = rng.permutation(n_samples)
    X = X[idx]
    y = y[idx]
    df = pd.DataFrame(X, columns=[f"sensor_{i}" for i in range(n_features)])
    df["label"] = y
    return df

if __name__ == '__main__':
    df = generate()
    df.to_csv("pm_synthetic.csv", index=False)
    print("Saved pm_synthetic.csv")
