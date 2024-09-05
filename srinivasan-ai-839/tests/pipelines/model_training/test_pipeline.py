"""
This is a boilerplate test file for pipeline 'model_training'
generated using Kedro 0.19.8.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from src.srinivasan_ai_839.pipelines.model_training.nodes import (
    split_dataset,
    train_model,
    evaluate_model,
)
from src.srinivasan_ai_839.pipelines.model_training.pipeline import (
    create_pipeline as create_pipeline_mode_train,
)
import pandas as pd
import typing as t
from sklearn.ensemble import RandomForestClassifier
import logging
from kedro.runner import SequentialRunner
import re
from kedro.io import DataCatalog
# from unittest.mock import MagicMock


def test_split_dataset(dummy_data):
    X_train, Y_train, X_test, Y_test = split_dataset(dummy_data)
    assert len(X_train) == 8
    assert len(Y_train) == 8
    assert len(X_test) == 2
    assert len(Y_test) == 2



# def test_model_training_pipeline(caplog):
#     pipeline = (create_pipeline_mode_train().from_nodes('split_data_node').to_nodes('evaluate_model_node'))
#     catalog = DataCatalog()
#     caplog.set_level(logging.DEBUG,logger="kedro")
#     successful_run_msg = f"Model has an accuracy of {.2f}"
#     SequentialRunner().run(pipeline,catalog)
#     assert successful_run_msg in caplog.text

