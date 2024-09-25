"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_dataset, train_model, evaluate_model, check_for_data_drift


def create_pipeline(**kwargs):
    """
    Create a machine learning pipeline with data splitting, model training, and evaluation nodes.

    The pipeline includes the following steps:
    1. Split the preprocessed data into training and testing sets.
    2. Check for data drift between training and testing sets.
    3. Train a RandomForestClassifier model using the training data.
    4. Give the nodes necessary tags such as training and inference based on the work the nodes do.

    
    Parameters:
    **kwargs: Additional keyword arguments for pipeline configuration.

    Returns:
    Pipeline: A Kedro pipeline object containing the defined nodes.
    """
    pipeline_train = pipeline(
        [
            node(
                func=split_dataset,
                inputs="preprocessed_data",
                outputs=["X_train", "X_test", "Y_train", "Y_test"],
                name="split_data_node",
                tags=["training"]
            ),
            node(
                func=check_for_data_drift,
                inputs=["Y_train", "Y_test"],
                outputs=None,
                name="data_drift_node",
                tags=["training"]
            ),
            node(
                func=train_model,
                inputs=["X_train", "Y_train"],
                outputs="model2",
                name="train_model_node",
                tags=["training"]
            ),
        ]
    )
    pipeline_inference = pipeline(
        [
                node(
                func=evaluate_model,
                inputs=["model2", "X_test", "Y_test"],
                outputs=None,
                name="evaluate_model_node",
                tags=["inference"]
            ),
        ]
    )
    return (pipeline_train+pipeline_inference)
