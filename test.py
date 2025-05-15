import requests
import joblib

model_bundle = joblib.load("emotion_model.pkl")
label_encoder = model_bundle["label_encoder"]


url = "http://localhost/mlflow-model/invocations"
headers = {
    "Content-Type": "application/json",
    "Host": "mlflow-model.local"
}
payload = {
    "instances": ["i am really tired of this nonsense"]
}

response = requests.post(url, json=payload, headers=headers)
numeric_preds = response.json()  # bijv. [3]

label_preds = label_encoder.inverse_transform(numeric_preds)

print("Model Prediction:", label_preds[0])