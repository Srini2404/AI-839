"""Srinivasan-ai-839"""

from .pipelines.data_preprocessing.nodes import _isTelephone, preprocess_data, preprocess_and_drift_checks,data_drift_quality_checks,data_drift_quality_checks_new

from .pipelines.model_training.nodes import evaluate_model, split_dataset, train_model

from .pipelines.data_preprocessing.pipeline import create_pipeline

from .pipelines.model_training.pipeline import create_pipeline as creat_pipeline1

__version__ = "0.1"

__all__ = [
    "_isTelephone",
    "preprocess_data",
    "evaluate_model",
    "split_dataset",
    "train_model",
    "create_pipeline",
    "create_pipeline1",
    "preprocess_and_drift_checks",
    "data_drift_quality_checks",
    "data_drift_quality_checks_new"
]
