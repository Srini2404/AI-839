"""
This is a boilerplate test file for pipeline 'model_training'
generated using Kedro 0.19.8.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from srinivasan_ai_839.pipelines.model_training.nodes import (
    check_for_data_drift,
    split_dataset,
    train_model,
    evaluate_model,
)
import pytest
import pandas as pd
import typing as t
from sklearn.ensemble import RandomForestClassifier
import logging
from kedro.runner import SequentialRunner
import re
from kedro.io import DataCatalog


def test_split_dataset(dummy_data):
    """
    Test the split_dataset function to ensure it correctly splits the dataset into training and testing sets.

    This test checks that the split_dataset function divides the input data into training and testing sets
    with the expected number of samples in each set.

    Args:
        dummy_data (pd.DataFrame): The input data to be split, provided by the dummy_data fixture.

    Raises:
        AssertionError: If the lengths of the training and testing sets do not match the expected values.
    """
    X_train, X_test, Y_train, Y_test = split_dataset(dummy_data)
    assert len(X_train) == 8, f"Expected 8 samples in X_train, got {len(X_train)}"
    assert len(Y_train) == 8, f"Expected 8 samples in Y_train, got {len(Y_train)}"
    assert len(X_test) == 2, f"Expected 2 samples in X_test, got {len(X_test)}"
    assert len(Y_test) == 2, f"Expected 2 samples in Y_test, got {len(Y_test)}"


def test_check_for_data_drift_no_drift(dummy_data_series):
    """
    Test to check if the function correctly identifies when there is no drift.

    Args:
        dummy_data_series (dict): A dictionary containing dummy data series for testing.
                                  The key "no_drift" should map to a tuple (y_train, y_test)
                                  where there is no data drift between y_train and y_test.

    Raises:
        pytest.fail: If an unexpected ValueError is raised indicating data drift.
    """
    y_train, y_test = dummy_data_series["no_drift"]

    try:
        check_for_data_drift(y_train, y_test)
        assert True  # No exception means the test passed
    except ValueError:
        pytest.fail("Unexpected drift detected when there should be none.")


def test_check_for_data_drift_with_drift(dummy_data_series):
    """
    Test to check if the function correctly identifies when there is drift.

    Args:
        dummy_data_series (dict): A dictionary containing dummy data series for testing.
                                  The key "drift" should map to a tuple (y_train, y_test)
                                  where there is data drift between y_train and y_test.

    Raises:
        pytest.raises: If a ValueError is not raised indicating data drift.
    """
    y_train, y_test = dummy_data_series["drift"]

    with pytest.raises(ValueError, match="Data drift detected"):
        check_for_data_drift(y_train, y_test)
