"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.7
"""

import pandas as pd


def _isTelephone(x:pd.Series) -> pd.Series:
    return x=="yes"

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    df['own_telephone'] = _isTelephone(df['own_telephone'])
    return df
