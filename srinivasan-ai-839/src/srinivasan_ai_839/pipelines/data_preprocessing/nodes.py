"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def _isTelephone(x:pd.Series) -> pd.Series:
    return x=="yes"

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    # convert all object columns to categorical values.
    df_processed = pd.get_dummies(df,drop_first=False)
    return df_processed

def _trainTestSplit(x:pd.DataFrame)-> tuple[pd.DataFrame,pd.DataFrame]:
    train,test = train_test_split(x,test_size=0.2)
    return train,test
