from pathlib import Path
import pandas as pd
import gzip
import json
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC, LinearSVC

data_path = Path(__file__).parent.parent / "files" / "input"


def clean_data(dataframe):
    """Limpia los datos de default de tarjeta de credito."""
    dataframe = dataframe.rename(
        columns={"default payment next month": "default"}
    )
    dataframe = dataframe.drop(
        columns=[
            column
            for column in dataframe.columns
            if column == "ID" or column.startswith("Unnamed:")
        ]
    )
    dataframe = dataframe.dropna()
    dataframe = dataframe[
        (dataframe["EDUCATION"] != 0) & (dataframe["MARRIAGE"] != 0)
    ]
    dataframe.loc[dataframe["EDUCATION"] > 4, "EDUCATION"] = 4
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


train_data = pd.read_csv(data_path / "train_data.csv.zip")
test_data = pd.read_csv(data_path / "test_data.csv.zip")

train_data = clean_data(train_data)
test_data = clean_data(test_data)

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

x_train = train_data.copy().drop(columns=["default"])
y_train = train_data["default"]

x_test = test_data.copy().drop(columns=["default"])
y_test = test_data["default"]

categorical_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

numeric_features = [
    column for column in x_train.columns if column not in categorical_features
]

preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("scaler", StandardScaler(), numeric_features)
        ],
        remainder="passthrough")

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", SVC(kernel="rbf",random_state=12345, max_iter=-1)),
])

from sklearn.metrics import (

        precision_score, 
        recall_score, 
        f1_score, 
        balanced_accuracy_score, 
        confusion_matrix,
    
    )
from sklearn.model_selection import GridSearchCV

params_grid =  {
    "pca__n_components": [20, x_train.shape[1] - 2],
    "selector__k": [12],
    "classifier__kernel": ["rbf"],
    "classifier__gamma": [0.1]
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1, 
    refit=True,
    verbose=1
)

model.fit(x_train, y_train)

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

models_path = Path(__file__).parent.parent / "files" / "models"
models_path.mkdir(parents=True, exist_ok=True)

with gzip.open(models_path / "model.pkl.gz", "wb") as file:
    pickle.dump(model, file)


output_path = Path(__file__).parent.parent / "files" / "output"
output_path.mkdir(parents=True, exist_ok=True)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


def calculate_metrics(dataset, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


metrics = [
    calculate_metrics("train", y_train, y_train_pred),
    calculate_metrics("test", y_test, y_test_pred),
]

with open(output_path / "metrics.json", "w", encoding="utf-8") as file:
    for metric in metrics:
        file.write(json.dumps(metric) + "\n")


def calculate_confusion_matrix(dataset, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(matrix[0][0]),
            "predicted_1": int(matrix[0][1]),
        },
        "true_1": {
            "predicted_0": int(matrix[1][0]),
            "predicted_1": int(matrix[1][1]),
        },
    }


confusion_matrices = [
    calculate_confusion_matrix("train", y_train, y_train_pred),
    calculate_confusion_matrix("test", y_test, y_test_pred),
]

with open(output_path / "metrics.json", "a", encoding="utf-8") as file:
    for matrix in confusion_matrices:
        file.write(json.dumps(matrix) + "\n")