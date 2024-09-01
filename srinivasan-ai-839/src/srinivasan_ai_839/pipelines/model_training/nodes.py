"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.8
"""
import logging
import pandas as pd;
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, r2_score

import typing as t


logger = logging.getLogger(__name__)

def split_dataset(dataset:pd.DataFrame) -> t.Tuple:
    dataset.drop(columns=["Unnamed: 0"],inplace=True)
    # dataset.columns
    X = dataset.copy()
    X.drop(columns="y",axis=1,inplace=True)
    Y = dataset['y']
    X_train,X_test, Y_train,Y_test = train_test_split(X,Y)
    return X_train,X_test,Y_train,Y_test



def train_model(X_train:pd.DataFrame,Y_train:pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier()
    model.fit(X_train,Y_train)
    return model


def evaluate_model(model:RandomForestClassifier,X_test:pd.DataFrame,Y_test:pd.Series):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    logger.info(f"Model has an accuracy of {accuracy} ")
