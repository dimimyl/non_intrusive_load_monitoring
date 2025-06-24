import pandas as pd
import numpy as np

def fill_empty_with_zero(df):
    """
    Fills empty spaces (NaN or blanks) in a DataFrame with zeros.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with empty spaces replaced by zeros.
    """
    # Replace NaN values with 0
    df_filled = df.fillna(0)

    # Replace empty strings or spaces with 0
    df_filled = df_filled.replace(r'^\s*$', 0, regex=True)

    return df_filled

def create_dataframe_from_csv(file_path):
    """
    Reads a CSV file and creates a DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The created DataFrame.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None


def rename_columns(df, new_names):
    """
    Renames all columns except the first two of a DataFrame based on the provided list of new names.

    Parameters:
        df (pd.DataFrame): The DataFrame whose columns are to be renamed.
        new_names (list): A list of new names for the columns except the first two.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns renamed.
    """
    n = len(new_names)
    if n != len(df.columns) - 2:
        raise ValueError("The number of new names must match the number of columns except the first two.")

    df.columns = list(df.columns[:2]) + new_names
    return df

def merge_ac_columns(df):
    # Step 1: Identify columns that start with 'ac'
    ac_columns = [col for col in df.columns if col.startswith('ac')]

    # Step 2: Sum the 'ac' columns row-wise and replace them with a single 'ac' column
    df['ac'] = df[ac_columns].sum(axis=1)

    # Step 3: Drop the original 'ac' columns from the DataFrame
    df.drop(columns=ac_columns, inplace=True)

    return df

def drop_columns(df, col_names):
    """
    Drops multiple columns from a DataFrame.

    Parameters:
    - df: The DataFrame
    - col_names: A list of column names to drop

    Returns:
    - A DataFrame with the columns dropped
    """
    # Check if all columns in the list exist in the DataFrame
    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not in the DataFrame: {missing_cols}")

    # Drop the columns and return the modified DataFrame
    return df.drop(columns=col_names)

# Prepare data for LSTM (Sliding window approach)
def create_sequences(input_data, target_data, seq_length):
    x, y = [], []
    for i in range(len(input_data) - seq_length):
        x.append(input_data[i:i + seq_length])
        y.append(target_data[i:i + seq_length])
    return np.array(x), np.array(y)