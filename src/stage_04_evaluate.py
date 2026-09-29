import os
import re
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

TEST_PATH = "C:/Users/Dell/Desktop/data/test.csv"
MODEL_PATH = "C:/Users/Dell/Desktop/data/xgb_model.json"

print("Loading test data and model for evaluation...")
test_df = pd.read_csv(TEST_PATH)

regex = re.compile(r"[\[\]<>]")
test_df.columns = [regex.sub("_", col) for col in test_df.columns]

X_test = test_df.drop(columns=["score"])
y_test = test_df["score"]

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

print("Calculating evaluation metrics...")
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n=== XGBoost Model Evaluation Results ===")
print(f"Test MSE  : {mse:.4f}")
print(f"Test RMSE : {rmse:.4f}")
print(f"Test MAE  : {mae:.4f}")
print(f"R2 Score  : {r2:.4f}\n")

scores = {"test_mse": float(mse), "test_rmse": float(rmse), "test_mae": float(mae), "r2_score": float(r2)}

with open("metrics.json", "w") as f:
    json.dump(scores, f, indent=4)