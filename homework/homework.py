from pathlib import Path
import gzip
import json
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


INPUT_PATH = Path(__file__).parent.parent / "files" / "input"
OUTPUT_PATH = Path(__file__).parent.parent / "files" / "output"
MODEL_PATH = Path(__file__).parent.parent / "files" / "models"

def clean_data(df):

    df = df.rename(
        columns={"default payment next month": "default"}
    )

    columns_to_drop = [
        c
        for c in df.columns
        if c == "ID" or c.startswith("Unnamed:")
    ]

    df = df.drop(columns=columns_to_drop)

    df = df.dropna()

    df = df.query("EDUCATION != 0 and MARRIAGE != 0")

    df["EDUCATION"] = df["EDUCATION"].clip(upper=4)

    return df.reset_index(drop=True)

def split_dataset(df):

    x = df.drop(columns="default")
    y = df["default"]

    return x, y

def build_pipeline(x_train):

    categorical = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    numeric = x_train.columns.difference(categorical).tolist()

    transformer = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", transformer),
            ("pca", PCA()),
            ("selector", SelectKBest(score_func=f_classif)),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    random_state=12345,
                ),
            ),
        ]
    )
    
def train_model(pipeline, x_train, y_train):

    parameters = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "selector__k": [12],
        "classifier__gamma": [0.1],
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    model.fit(x_train, y_train)

    return model

def save_model(model):

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    with gzip.open(MODEL_PATH / "model.pkl.gz", "wb") as file:
        pickle.dump(model, file)
        
def metrics_dict(name, y_true, y_pred):

    return {
        "type": "metrics",
        "dataset": name,
        "precision": float(precision_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


def confusion_dict(name, y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {
            "predicted_0": int(tn),
            "predicted_1": int(fp),
        },
        "true_1": {
            "predicted_0": int(fn),
            "predicted_1": int(tp),
        },
    }

def save_results(results):

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_PATH / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:
            file.write(json.dumps(result))
            file.write("\n")

def pregunta_01():

    train = clean_data(pd.read_csv(INPUT_PATH / "train_data.csv.zip"))
    test = clean_data(pd.read_csv(INPUT_PATH / "test_data.csv.zip"))

    x_train, y_train = split_dataset(train)
    x_test, y_test = split_dataset(test)

    pipeline = build_pipeline(x_train)

    model = train_model(
        pipeline,
        x_train,
        y_train,
    )

    save_model(model)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    results = [
        metrics_dict("train", y_train, train_pred),
        metrics_dict("test", y_test, test_pred),
        confusion_dict("train", y_train, train_pred),
        confusion_dict("test", y_test, test_pred),
    ]

    save_results(results)