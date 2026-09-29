import os
import pandas as pd
from river import linear_model, preprocessing

if os.path.exists("drift_status.txt"):
    with open("drift_status.txt", "r") as f:
        status = f.read().strip()

if status == "DRIFT_DETECTED":
    print("Drift found! Triggering Online Learning Update...")

    scaler = preprocessing.StandardScaler()
    model = linear_model.LinearRegression()

    data = pd.read_csv("test.csv")

    for index, row in data.iterrows():
        X = row.iloc[:-1].to_dict()
        y = row.iloc[-1]
        model.learn_one(X, y)

    print("Model updated with new data stream using Online Learning!")
else:
    print("No drift detected . Online learning update not required.")