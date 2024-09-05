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

# Configure logger
logger = logging.getLogger(__name__)


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
    Trains a RandomForestClassifier model using the provided training data.

    Parameters:
    X_train (pd.DataFrame): The training features.
    Y_train (pd.Series): The training target.

    Returns:
    RandomForestClassifier: The trained RandomForestClassifier model.
    """
    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, Y_train)

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

    Returns:
    None
    """
    # Predict the target for test data
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    # Log the accuracy
    logger.info(f"Model has an accuracy of {accuracy:.2f}")
