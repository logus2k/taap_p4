import numpy as np


def make_windows(data, target_idx, lookback=120, horizon=24):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, target_idx])
    return np.array(X), np.array(y)