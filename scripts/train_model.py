import pandas as pd
import argparse
from sklearn.base import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset_train', type=str, required=True)
parser.add_argument('--input_dataset_validation', type=str, required=True)
parser.add_argument('--input_dataset_test', type=str, required=True)
parser.add_argument('--output_model', type=str, required=True)
args = parser.parse_args()

# MLFlow setup:
MLFLOW_TRACKING_URI = "http://istio-ingressgateway.istio-system.svc.cluster.local/mlflow/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Creating an MLFlow experiment
experiment_name = "mood-machine-experiment"
try:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location="s3://ml-models/"
    )
except mlflow.exceptions.MlflowException:
    pass

print("MLflow Tracking URI:", mlflow.get_tracking_uri())


# Load our datasets from the arguments, taken from MinIO in previous pipeline step.
df_train = pd.read_csv(args.input_dataset_train)
df_val = pd.read_csv(args.input_dataset_validation)
df_test = pd.read_csv(args.input_dataset_test)

# Encoding our labels, ML models work with numeric labels.
# We transform our string labels to numeric labels using the LabelEncoder()
# joy -> 1
# sadness -> 3
le = LabelEncoder()
y_train = le.fit_transform(df_train['label'])
y_val = le.transform(df_val['label'])
y_test = le.transform(df_test['label'])

with mlflow.start_run() as run:
# Training pipeline
# We are using a TfidVectorizer to transorm our text to numeric values
# Text: "I feel sad" -> Vector: [0, 0.4, 0.1, 0, 0.9]
    pipeline = Pipeline([
      ('tfidf', TfidfVectorizer(max_features=5000)),
      ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(df_train['text'], y_train)

    # Evaluate our model on our test set
    y_pred = pipeline.predict(df_test["text"])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv")
    mlflow.log_artifact("classification_report.csv")

    # Save combined model + label encoder
    model_bundle = {'model': pipeline, 'label_encoder': le}
    joblib.dump(model_bundle, args.output_model)

    # Log the model as an MLflow model
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="mood-machine-model",
        input_example=[df_train["text"].iloc[0]]
    )