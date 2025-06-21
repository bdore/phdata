import requests
import json
import pandas as pd


examples_df = pd.read_csv("data/future_unseen_examples.csv", dtype={"zipcode": str})
predict_data = examples_df.to_dict(orient="records")

url = "http://localhost:8000/predictions/all"
# data = json.dumps([predict_data[0]])  # Uncomment to test with a single example.
data = json.dumps(predict_data)
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=data, headers=headers)

if response.status_code == 200:
    predictions = response.json()
    print(json.dumps(predictions, indent=4))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
