"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_dataset, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a machine learning pipeline.

    This pipeline consists of three nodes:
    1. Splitting the preprocessed data into training and testing sets.
    2. Training a RandomForestClassifier model using the training data.
    3. Evaluating the trained model using the testing data.

    Parameters:
    **kwargs: Additional keyword arguments.

    Returns:
    Pipeline: A Kedro pipeline object with the defined nodes.
    """
    pipeline_instance = pipeline(
        [
            node(
                func=split_dataset,
                inputs="preprocessed_data",
                outputs=["X_train", "X_test", "Y_train", "Y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "Y_train"],
                outputs="classifier_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier_model", "X_test", "Y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
    return pipeline_instance
