import numpy as np

def pad_signal_for_sequences(signal, seq_length, step):
    original_length = len(signal)
    remainder = (original_length - seq_length) % step
    if remainder != 0:
        pad_len = step - remainder
        padding = np.zeros((pad_len, signal.shape[1]))
        signal = np.vstack([signal, padding])
    return signal, original_length

def create_sequences_wp(X, y, seq_length, step):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - seq_length + 1, step):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i:i+seq_length])
    return np.array(X_seq), np.array(y_seq)


def revert_sequences_wp(sequences, seq_length, step):
    num_samples, _, num_features = sequences.shape
    original_length = (num_samples - 1) * step + seq_length
    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))

    for i in range(num_samples):
        start = i * step
        end = start + seq_length
        reconstructed[start:end] += sequences[i]
        counts[start:end] += 1

    counts[counts == 0] = 1
    return reconstructed / counts