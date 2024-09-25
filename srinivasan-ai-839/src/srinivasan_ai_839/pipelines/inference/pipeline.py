"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline,node
from .nodes import pre_processing,run_inference

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=pre_processing,
            inputs=["inference_data","encoder"],
            outputs="inference_processed_data",      
            name="data_preprocess",
            # tags=[""]
        ),
        node(
            func=run_inference,
            inputs=["model_3","inference_processed_data"],
            outputs="output_result"
        )
    ])
