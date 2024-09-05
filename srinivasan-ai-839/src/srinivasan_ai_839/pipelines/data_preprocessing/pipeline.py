"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, pipeline, node

# from .nodes import preprocess_data

from .nodes import preprocess_and_drift_checks

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a data processing pipeline.

    This pipeline consists of a single node that preprocesses the data from the input dataset.

    Parameters:
    **kwargs: Additional keyword arguments.

    Returns:
    Pipeline: A Kedro pipeline object with the defined nodes.
    """
    return pipeline(
        [
            node(
                func=preprocess_and_drift_checks,
                inputs="dataset_id_214",
                outputs=["preprocessed_data","data_drift_plot","target_plot"],
                name="preprocessed_data_node",
            )
        ]
    )
