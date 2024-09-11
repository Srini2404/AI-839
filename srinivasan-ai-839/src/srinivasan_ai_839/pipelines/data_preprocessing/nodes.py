"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns,
    TestShareOfMissingValues,
    # Add this line to import the missing class
)
from evidently import ColumnMapping
import plotly.graph_objects as go

import json

import plotly.express as px


def _isTelephone(x: pd.Series) -> pd.Series:
    """
    Checks if the values in the given pandas Series are equal to "yes".

    Parameters:
    x (pd.Series): A pandas Series containing the data to be checked.

    Returns:
    pd.Series: A pandas Series of boolean values where True indicates the value is "yes" and False otherwise.
    """
    return x == "yes"


def preprocess_and_drift_checks(df: pd.DataFrame):
    """
    Preprocesses the data and performs data drift quality checks.

    Args:
        df: A pandas DataFrame containing the data to be processed and checked for drift.

    Returns:
        tuple: A tuple containing the preprocessed data DataFrame, drift plot, and target plot.
    """ 
    preprocessed_data = preprocess_data(df)
    print("Hello World!")
    target_plot = data_drift_quality_checks_new(df)
    drift_plot = data_drift_quality_checks(df)

    return preprocessed_data, drift_plot, target_plot


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
    df_processed.to_csv("C:\\Users\\Admin\\Desktop\\Semester_7\\MLOps\\AI-839\\srinivasan-ai-839\\data\\01_raw\\pre_processed.csv",index=False)
    return df_processed


def data_drift_quality_checks(df: pd.DataFrame) -> go.Figure:
    """
    Performs data drift quality checks on the given DataFrame and returns a plotly figure.

    Args:
        df: A pandas DataFrame containing the data to be checked for drift.

    Returns:
        go.Figure: A plotly figure representing the data drift quality checks.
    """
    # Define the test suite
    tests = TestSuite(
        tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ]
    )

    # Sample reference and current data
    references = df.sample(n=50)
    current = df.sample(n=50)

    # Run tests
    tests.run(reference_data=references, current_data=current)
    results = tests.as_dict()

    # Extract test names and statuses
    test_names = [test["name"] for test in results["tests"]]
    test_statuses = [
        "SUCCESS" if test["status"] == "SUCCESS" else "FAILURE"
        for test in results["tests"]
    ]

    # Define colors for the plot (green for success, red for failure)
    colors = ["green" if status == "SUCCESS" else "red" for status in test_statuses]

    # Plot Results using Plotly
    fig = go.Figure()

    # Add bars for each test result
    fig.add_trace(
        go.Bar(
            x=test_names,
            y=[1]
            * len(
                test_names
            ),  # Set y value to 1 for all tests (since we're focusing on status)
            marker=dict(color=colors),  # Apply color based on status
            text=test_statuses,  # Add status as text labels
            hoverinfo="text",  # Show status on hover
            showlegend=False,
        )
    )

    # Set up the layout
    fig.update_layout(
        title="Data Quality and Drift Test Results",
        xaxis_title="Test Name",
        yaxis_title="Test Status",
        yaxis=dict(tickvals=[1], ticktext=[""]),
        xaxis_tickangle=-45,
    )
    # Return the figure object for further use in Kedrofig.show()
    return fig


def data_drift_quality_checks_new(df: pd.DataFrame) -> go.Figure:
    """
    Performs new data drift quality checks on the given DataFrame and returns a plotly figure.

    Args:
        df: A pandas DataFrame containing the data to be checked for drift.

    Returns:
        go.Figure: A plotly figure representing the new data drift quality checks.
    """
    # Define the test suite with additional tests
    tests = TestSuite(
        tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestShareOfMissingValues(),
            TestNumberOfDriftedColumns(),
        ]
    )

    # Sample reference and current data
    references = df.sample(n=50)
    current = df.sample(n=50)

    # Run tests
    tests.run(reference_data=references, current_data=current)
    results = tests.as_dict()

    # Categorize tests into success, warning, and failure
    test_statuses = [test["status"] for test in results["tests"]]
    success_count = test_statuses.count("SUCCESS")
    failure_count = test_statuses.count("FAIL")
    warning_count = len(test_statuses) - success_count - failure_count

    # Define categories and counts for the stacked bar chart
    categories = ["Success", "Warning", "Failure"]
    values = [success_count, warning_count, failure_count]

    # Create the Plotly figure (stacked bar chart)
    fig = go.Figure()

    # Add bars for each status category
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=["green", "orange", "red"]
            ),  # Different colors for each category
            text=values,
            textposition="auto",
            hoverinfo="text",
        )
    )

    # Set up the layout
    fig.update_layout(
        title="Data Quality and Drift Test Results",
        xaxis_title="Test Status",
        yaxis_title="Number of Tests",
        yaxis=dict(
            tickvals=list(
                range(max(values) + 1)
            )  # Adjust the y-axis ticks based on max value
        ),
        barmode="stack",
    )

    # Return the figure object for further use in Kedro
    return fig
