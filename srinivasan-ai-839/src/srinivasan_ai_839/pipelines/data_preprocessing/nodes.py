"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def _isTelephone(x: pd.Series) -> pd.Series:
    """
    Checks if the values in the given pandas Series are equal to "yes".

    Parameters:
    x (pd.Series): A pandas Series containing the data to be checked.

    Returns:
    pd.Series: A pandas Series of boolean values where True indicates the value is "yes" and False otherwise.
    """
    return x == "yes"

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object columns in the given DataFrame to categorical values using one-hot encoding.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing the data to be processed.

    Returns:
    pd.DataFrame: A pandas DataFrame with object columns converted to categorical values.
    """
    # convert all object columns to categorical values.
    df_processed = pd.get_dummies(df, drop_first=False)
    return df_processed

