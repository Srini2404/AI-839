"""
This module contains tests for the data preprocessing pipeline.
It includes tests for various data quality checks and the preprocessing function.
"""

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns,
)
from evidently import ColumnMapping
import plotly.graph_objects as go

# from kedro.framework.session.session import get_current_session
import pandas as pd
from srinivasan_ai_839.pipelines.data_preprocessing.nodes import (
    _isTelephone,
    preprocess_data,
)
import logging
from srinivasan_ai_839.pipelines.data_preprocessing import (
    create_pipeline as create_pipeline_dp,
)
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

def test_preprocess_data(dummy_data):
    """
    Test the preprocess_data function to ensure it correctly transforms
    all columns to numerical types.

    Args:
        dummy_data (pd.DataFrame): The input data to preprocess.

    Raises:
        AssertionError: If any column in the processed data is of type 'object'.
    """
    processed_data = preprocess_data(dummy_data)
    for col in processed_data.columns:
        assert (
            processed_data[col].dtype != "object"
        ), f"Column {col} is of type object, but it should be transformed to numerical type."
