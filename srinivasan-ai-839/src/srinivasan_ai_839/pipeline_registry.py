"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from srinivasan_ai_839 import __version__ as PROJECT_VERSION
from kedro_mlflow.pipeline import pipeline_ml_factory
from srinivasan_ai_839.pipelines.data_preprocessing.pipeline import create_pipeline as create_pipeline_dp
from srinivasan_ai_839.pipelines.model_training.pipeline import create_pipeline as create_pipeline_mt

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    # data_process_pipeline = create_pipeline_dp()
    # model_pipeline = create_pipeline_mt()
    # inference_pip = model_pipeline.only_nodes_with_tags("inference")
    # training_pipeline_ml = pipeline_ml_factory(
    #     training = model_pipeline.only_nodes_with_tags("training"),
    #     inference = inference_pip,
    #     input_name="preprocessed_data",
    #     log_model_kwargs = dict(
    #         artifact_path="srinivasan_ai_839",
    #         # conda_env={
    #         #         "python": 3.10,
    #         #         "build_dependencies": ["pip"],
    #         #         "dependencies": [f"kedro_mlflow_tutorial=={PROJECT_VERSION}"],
    #         # },
    #         signature="auto",
    #         ),
    #     )
   
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
    # return {
        # "data_proprocessing":data_process_pipeline,
        # "training": training_pipeline_ml + pipelines
    # }
