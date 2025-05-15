import requests

url = "http://localhost/mlflow-model/invocations"
headers = {
    "Content-Type": "application/json",
    "Host": "mlflow-model.local"
}
payload = {
    "instances": ["i really cant stop smiling"]
}

response = requests.post(url, json=payload, headers=headers)
print("Model Prediction:", response.json())
