import pandas as pd

def split_dataset(df, testing_days=2, val_split=0.4):
    """
    Splits a time-indexed DataFrame into training, validation, and testing sets.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a DateTimeIndex.
        testing_days (int): Number of days at the end for testing.
        val_split (float): Percentage (0–1) of the remaining data for validation.
    
    Returns:
        train_data (pd.DataFrame), val_data (pd.DataFrame), test_data (pd.DataFrame)
    """
    df = df.sort_index()

    # Define test range
    last_timestamp = df.index.max()
    first_testing_time = last_timestamp - pd.Timedelta(days=testing_days)

    # Split into testing and the rest
    df_testing = df[df.index > first_testing_time]
    df_rest = df[df.index <= first_testing_time]

    # Split the remaining into training and validation
    n_total = len(df_rest)
    n_val = int(n_total * val_split)

    if n_val == 0:
        df_validation = df_rest.iloc[0:0]
        df_training = df_rest
    else:
        df_validation = df_rest.iloc[-n_val:]
        df_training = df_rest.iloc[:-n_val]

    return df_training, df_validation, df_testing
