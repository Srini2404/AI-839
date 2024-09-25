"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""
import pandas as pd

import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import logging
import os

log_file = "model_usage.log"

# Configure logging to log everything to the file, including DEBUG level messages
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

def run_inference(model: GradientBoostingClassifier, data: pd.DataFrame):
    try:
        logging.info("Starting inference process...")
        df = data
        prediction = model.predict(df)
        predictions = pd.DataFrame(prediction)
        
        log_usage(model, df, predictions)
        
        logging.info("Inference completed.")
        return predictions
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise


def log_usage(model: GradientBoostingClassifier, input_data: pd.DataFrame, predictions: pd.DataFrame):
    """
    Logs details about model usage including input data, predictions, model details, and timestamp.
    """
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Number of trees in the model: {model.n_estimators}")
    logging.info(f"Model Parameters: {model.get_params()}")
    logging.info(f"Input shape: {input_data.shape}")
    logging.debug(f"Input data:\n{input_data.head()}")
    logging.info(f"Output shape: {predictions.shape}")
    logging.debug(f"Predictions:\n{predictions.head()}")
    logging.info(f"Timestamp: {datetime.datetime.now()}")
