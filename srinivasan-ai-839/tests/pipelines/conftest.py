import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="package")
def dummy_data():
    """
    Fixture that provides a dummy DataFrame for testing purposes.

    The DataFrame contains the following columns and more:
    - 'Unnamed: 0': Index column with integer values.
    - 'checking_status': Categorical column with various checking account statuses.
    - 'duration': Integer column representing the duration of the credit in months.
    - 'credit_history': Categorical column with different credit history statuses.

    Returns:
        pd.DataFrame: A DataFrame with dummy data for testing.
    """
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "checking_status": [
                "no checking",
                "no checking",
                "0<=X<200",
                "no checking",
                "no checking",
                "0<=X<200",
                "<0",
                "<0",
                "no checking",
                "no checking",
            ],
            "duration": [24, 36, 12, 48, 15, 30, 18, 9, 24, 48],
            "credit_history": [
                "existing paid",
                "critical/other existing credit",
                "delayed previously",
                "existing paid",
                "existing paid",
                "critical/other existing credit",
                "all paid",
                "no credits/all paid",
                "existing paid",
                "critical/other existing credit",
            ],
            "purpose": [
                "furniture/equipment",
                "new car",
                "radio/tv",
                "education",
                "furniture/equipment",
                "used car",
                "radio/tv",
                "new car",
                "new car",
                "used car",
            ],
            "credit_amount": [
                1500,
                2500,
                1000,
                3000,
                2186,
                4000,
                600,
                1200,
                1800,
                2500,
            ],
            "savings_status": [
                "<100",
                "100<=X<500",
                "no known savings",
                "<100",
                "no known savings",
                "100<=X<500",
                "<100",
                "no known savings",
                "<100",
                "no known savings",
            ],
            "employment": [
                "7<=X<10",
                "1<=X<4",
                "4<=X<7",
                "7<=X<10",
                "4<=X<7",
                "4<=X<7",
                "7<=X<10",
                "1<=X<4",
                "4<=X<7",
                "unemployed",
            ],
            "installment_commitment": [3, 2, 4, 2, 1, 3, 2, 1, 2, 3],
            "personal_status": [
                "male single",
                "female div/dep/mar",
                "female div/dep/mar",
                "male single",
                "female div/dep/mar",
                "male single",
                "female div/dep/mar",
                "male single",
                "male single",
                "female div/dep/mar",
            ],
            "other_parties": [
                "none",
                "guarantor",
                "none",
                "guarantor",
                "none",
                "none",
                "none",
                "none",
                "co applicant",
                "guarantor",
            ],
            "residence_since": [2, 3, 4, 1, 4, 2, 3, 1, 2, 4],
            "property_magnitude": [
                "real estate",
                "life insurance",
                "no known property",
                "real estate",
                "real estate",
                "life insurance",
                "car",
                "life insurance",
                "real estate",
                "no known property",
            ],
            "age": [45, 34, 28, 32, 33, 38, 40, 27, 29, 30],
            "other_payment_plans": [
                "none",
                "none",
                "bank",
                "none",
                "bank",
                "none",
                "none",
                "none",
                "none",
                "none",
            ],
            "housing": [
                "own",
                "own",
                "rent",
                "own",
                "rent",
                "own",
                "rent",
                "own",
                "own",
                "rent",
            ],
            "existing_credits": [2, 1, 1, 1, 1, 2, 1, 1, 2, 2],
            "job": [
                "skilled",
                "unskilled resident",
                "skilled",
                "high qualif/self emp/mgmt",
                "unskilled resident",
                "skilled",
                "skilled",
                "skilled",
                "skilled",
                "unskilled resident",
            ],
            "num_dependents": [2, 1, 2, 1, 1, 2, 2, 1, 1, 1],
            "own_telephone": [
                "none",
                "none",
                "yes",
                "none",
                "none",
                "yes",
                "none",
                "none",
                "yes",
                "yes",
            ],
            "foreign_worker": [
                "no",
                "yes",
                "yes",
                "yes",
                "yes",
                "yes",
                "no",
                "yes",
                "no",
                "yes",
            ],
            "health_status": [
                "good",
                "good",
                "good",
                "excellent",
                "good",
                "excellent",
                "good",
                "good",
                "good",
                "good",
            ],
            "X_1": [
                6.234567,
                3.456789,
                2.345678,
                4.567891,
                4.824403,
                5.123456,
                3.567890,
                2.678901,
                5.678912,
                4.567890,
            ],
            "X_2": [
                732.456789,
                123.456789,
                234.567890,
                789.012345,
                706.7856345,
                820.345678,
                456.789012,
                345.678901,
                912.345678,
                678.901234,
            ],
            "X_3": [
                0.987654,
                0.876543,
                0.765432,
                0.654321,
                0.9097408,
                0.8765432,
                0.8765432,
                0.6543210,
                0.9876543,
                0.7654321,
            ],
            "X_4": [
                5.012345,
                4.567890,
                3.456789,
                6.234567,
                5.05123738,
                5.678901,
                4.567890,
                3.456789,
                5.789012,
                4.567890,
            ],
            "X_5": [
                0.06234567,
                0.04567890,
                0.03456789,
                0.07890123,
                0.04824402824,
                0.06789012,
                0.05678901,
                0.04567890,
                0.07890123,
                0.05678901,
            ],
            "X_6": [
                0.3456789,
                0.4567890,
                0.5678901,
                0.6789012,
                0.3557247209,
                0.5678901,
                0.6789012,
                0.5678901,
                0.6789012,
                0.5678901,
            ],
            "X_7": [
                0.8765432,
                0.6543210,
                0.5432109,
                0.4321098,
                0.9097408001,
                0.6543210,
                0.7654321,
                0.5432109,
                0.6543210,
                0.5432109,
            ],
            "X_8": [
                0.4012345,
                0.5012345,
                0.6012345,
                0.7012345,
                0.405123738,
                0.5012345,
                0.6012345,
                0.5012345,
                0.6012345,
                0.5012345,
            ],
            "X_9": [
                0.2123456,
                0.3123456,
                0.4123456,
                0.5123456,
                0.2607669503,
                0.3123456,
                0.4123456,
                0.3123456,
                0.4123456,
                0.3123456,
            ],
            "X_10": [
                0.1456789,
                0.2456789,
                0.3456789,
                0.4456789,
                0.1036180293,
                0.2456789,
                0.3456789,
                0.2456789,
                0.3456789,
                0.2456789,
            ],
            "X_11": [
                0.4012345,
                0.3012345,
                0.2012345,
                0.1012345,
                0.4063758137,
                0.3012345,
                0.2012345,
                0.3012345,
                0.2012345,
                0.3012345,
            ],
            "X_12": [
                0.02456789,
                0.01456789,
                0.03456789,
                0.05456789,
                0.01789696024,
                0.01456789,
                0.03456789,
                0.01456789,
                0.03456789,
                0.01456789,
            ],
            "X_13": [
                0.5123456,
                0.6123456,
                0.7123456,
                0.8123456,
                0.5186819042,
                0.6123456,
                0.7123456,
                0.6123456,
                0.7123456,
                0.6123456,
            ],
            "y": [True, False, True, True, True, False, True, False, True, False],
        }
    )


@pytest.fixture(scope="package")
def dummy_data_series():
    """
    Fixture that provides dummy data series for testing data drift detection.

    This fixture creates two sets of dummy data:
    - One set where there is no drift between the training and testing data.
    - One set where there is a significant drift between the training and testing data.

    Returns:
        dict: A dictionary with two keys:
            - "no_drift": A tuple (y_train_no_drift, y_test_no_drift) where both series have similar distributions.
            - "drift": A tuple (y_train_drift, y_test_drift) where the series have different distributions indicating drift.
    """
    # No drift (similar distributions)
    y_train_no_drift = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
    y_test_no_drift = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])

    # Drift (different distributions)
    y_train_drift = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # Mostly 0s
    y_test_drift = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # Mostly 1s

    return {
        "no_drift": (y_train_no_drift, y_test_no_drift),
        "drift": (y_train_drift, y_test_drift),
    }


# @pytest.fixture(scope="package")
# def train_x():
#     return X_train


# @pytest.fixture(scope="package")
# def train_y():
#     return Y_train


# @pytest.fixture(scope="package")
# def test_x():
#     return X_test


# @pytest.fixture(scope="package")
# def test_y():
#     return Y_test
