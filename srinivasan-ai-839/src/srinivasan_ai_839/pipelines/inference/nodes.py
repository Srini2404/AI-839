"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


def pre_processing(data: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Transforms the categorical columns of the input DataFrame using a pre-fitted OneHotEncoder.
    The transformed categorical columns are concatenated with the original numeric columns.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing categorical columns to be transformed.
    encoder : OneHotEncoder
        A pre-fitted OneHotEncoder instance for transforming the categorical columns.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where the categorical columns have been one-hot encoded.
        The original numerical columns are retained and concatenated with the transformed columns.
    """
    # Identify numeric and categorical columns
    num_cols = data._get_numeric_data()
    cat_cols = data.select_dtypes(include=['object'])

    if not cat_cols.empty:
        # Use the provided encoder to transform the categorical columns
        cat_cols_ohe = encoder.transform(cat_cols)
        
        # Convert the one-hot encoded result into a DataFrame
        cat_cols_ohe_df = pd.DataFrame(
            cat_cols_ohe, columns=encoder.get_feature_names_out(cat_cols.columns)
        )
        
        # Concatenate the one-hot encoded categorical columns with the numeric columns
        data_processed = pd.concat([cat_cols_ohe_df, num_cols.reset_index(drop=True)], axis=1)
    else:
        data_processed = data

    return data_processed


def run_inference(model:RandomForestClassifier, data:pd.DataFrame) :
    df = data
    prediction = model.predict(df)
    print(prediction)
    return pd.DataFrame(prediction)


