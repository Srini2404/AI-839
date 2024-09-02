"""
This is a boilerplate test file for pipeline 'model_training'
generated using Kedro 0.19.8.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from src.srinivasan_ai_839.pipelines.model_training.nodes import split_dataset, train_model,evaluate_model
import pandas as pd
import typing as t
from sklearn.ensemble import RandomForestClassifier
import logging

def test_split_dataset(dataset:pd.DataFrame) -> t.Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    X_train,Y_train,X_test,Y_test = split_dataset(dataset=dataset)
    assert len(X_train) ==8
    assert len(Y_train) ==8
    assert len(X_test) == 2
    assert len(Y_test) == 2




