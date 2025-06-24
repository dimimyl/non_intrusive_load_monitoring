import pandas as pd
import numpy as np

def revert_sequences(sequences, seq_length):
    """
    Revert the sequence creation process to reconstruct the original signal.

    Args:
        sequences (numpy array): Predicted or ground truth sequences,
                                 shape (num_samples, seq_length, num_features).
        seq_length (int): Length of each sequence window.

    Returns:
        numpy array: Reconstructed original signal, shape (original_length, num_features).
    """
    # Initialize an array to store the reconstructed signal
    num_samples, _, num_features = sequences.shape
    original_length = num_samples + seq_length - 1
    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))  # Track the number of overlaps at each position

    for i in range(num_samples):
        reconstructed[i:i + seq_length] += sequences[i]
        counts[i:i + seq_length] += 1

    # Average overlapping regions
    reconstructed /= counts
    return reconstructed

def save_to_csv(y_test, y_pred):
    """
    Save y_test and y_pred to CSV files.

    Args:
        y_test (numpy array): Rescaled ground truth values.
        y_pred (numpy array): Rescaled predicted values.
    """
    y_test_df = pd.DataFrame(y_test, columns= ['st', 'wh', 'dw', 'kettler', 'wm', 'toaster', 'fridge'])
    y_pred_df = pd.DataFrame(y_pred, columns= ['st', 'wh', 'dw', 'kettler', 'wm', 'toaster', 'fridge'])

    y_test_df.to_csv("csv_files/real_values.csv", index=False)
    y_pred_df.to_csv(f"csv_files/predictions.csv", index=False)
    print(f"Files saved as real_values.csv and predictions.csv")

def redd_accuracy (y_true, y_pred):
    abs_errors = np.abs(y_pred - y_true)
    numerator = np.sum(abs_errors)
    denominator = 2 * np.sum(np.sum(y_true, axis=1))  # 2 * sum of total power over all time steps
    accuracy = 1 - (numerator / denominator)
    return accuracy