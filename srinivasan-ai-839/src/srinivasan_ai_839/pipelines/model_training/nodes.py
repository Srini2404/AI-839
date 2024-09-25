"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import typing as t
from evidently import ColumnMapping
import os
import mlflow
from mlflow import sklearn
from datetime import datetime
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Configure logger
logger = logging.getLogger(__name__)


def check_for_data_drift(y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Checks for data drift between training and testing target variables using Evidently.

    Parameters:
    y_train (pd.Series): The target variable from the training set.
    y_test (pd.Series): The target variable from the testing set.

    Raises:
    ValueError: If drift is detected between the distributions of y_train and y_test.
    """
    # Create a DataFrame with the target columns
    df_train = pd.DataFrame({"y": y_train})
    df_test = pd.DataFrame({"y": y_test})

    # Define column mapping for Evidently
    column_mapping = ColumnMapping(target="y")

    # Create a data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(
        reference_data=df_train, current_data=df_test, column_mapping=column_mapping
    )

    # Extract the drift report
    drift_report = report.as_dict()
    # Check if drift was detected
    drift_detected = drift_report["metrics"][0]["result"]["drift_by_columns"]["y"][
        "drift_detected"
    ]
    # print("drift_detected_metrics")
    if drift_detected:
        raise ValueError(
            "Data drift detected between training and testing target variables."
        )
    else:
        print("No data drift detected.")


def split_dataset(
    dataset: pd.DataFrame,
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets for features and target.

    Parameters:
    dataset (pd.DataFrame): The dataset to be split.

    Returns:
    t.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the training features, testing features, training target, and testing target.
    """
    # Drop the unnecessary column
    dataset.drop(columns=["Unnamed: 0"], inplace=True)

    # Separate features and target
    X = dataset.drop(columns="y")
    Y = dataset["y"]

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    return X_train, X_test, Y_train, Y_test


def train_model(X_train: pd.DataFrame, Y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier model using the provided training data. Used Kedro-MlFlow plugin to log the artifact.

    Parameters:
    X_train (pd.DataFrame): The training features.
    Y_train (pd.Series): The training target.

    Returns:
    RandomForestClassifier: The trained RandomForestClassifier model.
    """
    # Initialize the model

    model = RandomForestClassifier(random_state=42)
    mlflow.log_artifact("C:\\Users\\Admin\\Desktop\\Semester_7\\MLOps\\AI-839\\srinivasan-ai-839\\data\\01_raw\\dataset_id_214.csv")
    mlflow.autolog()
    mlflow.log_artifact(local_path=os.path.join("data", "02_modelinput", "preprocessed_data.csv"))
    # Train the model
    model.fit(X_train, Y_train)
    # model = np.vstack(weights).transpose()
    sklearn.log_model(sk_model=model, artifact_path="model")
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, Y_test: pd.Series
):
    """
    Evaluates the trained model using the testing data and logs the accuracy.

    Parameters:
    model (RandomForestClassifier): The trained model to be evaluated.
    X_test (pd.DataFrame): The testing features.
    Y_test (pd.Series): The testing target.

    Used Kedro mlflow plugin to log the accuracy , the time of prediction.

    Returns:
    None
    """
    # Predict the target for test data
    Y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
    mlflow.set_tag("Initial kedro-mlfow","9th September")
    # Log the accuracy
    # logger.info(f"Model has an accuracy of {accuracy:.2f}")
    print(f"The accuracy score is: {accuracy}")
