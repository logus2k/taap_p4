import numpy as np


def make_timegan_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


def split_synthetic_sequence(sequence, lookback, horizon, target_idx):
    X = sequence[:, :lookback, :]
    y = sequence[:, lookback:lookback + horizon, target_idx]
    return X, y


def split_synthetic_sequences(sequences, lookback, horizon, target_idx):
    X = sequences[:, :lookback, :]
    y = sequences[:, lookback:lookback + horizon, target_idx]
    return X, y