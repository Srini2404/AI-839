"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""
import pandas as pd

import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import logging
import os

log_file = "model_usage.log"

# Ensure the log file is created if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, 'w'):  # This will create the file if it doesn't exist
        pass
print(os.curdir)

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,  # Set to DEBUG to capture everything
    format="%(asctime)s - %(levelname)s - %(message)s",
)



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
    # print(prediction)
    log_usage(model,df,prediction)
    return pd.DataFrame(prediction)



def log_usage(model: RandomForestClassifier, input_data: pd.DataFrame, predictions: pd.DataFrame):
    """
    Logs details about model usage including input data, predictions, model details, and timestamp.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        The trained model being used for inference.
    input_data : pd.DataFrame
        The input data for inference.
    predictions : pd.DataFrame
        The predictions made by the model.
    """
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Number of trees in the model: {model.n_estimators}")
    logging.info(f"Model Parameters: {model.get_params()}")
    logging.info(f"Input shape: {input_data.shape}")
    logging.debug(f"Input data:\n{input_data.head()}")
    logging.info(f"Output shape: {predictions.shape}")
    logging.debug(f"Predictions:\n{predictions.head()}")
    logging.info(f"Timestamp: {datetime.now()}")

