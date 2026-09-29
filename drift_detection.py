import pandas as pd
from river import drift

data = pd.read_csv("test.csv")

drift_detector = drift.ADWIN()
drift_detected = False

print("Checking for Concept Drift")

for index, row in data.iterrows():
    val = row.iloc[0]
    drift_detector.update(val)

    if drift_detector.drift_detected:
        print(f"Drift detected at row {index}!")
        drift_detected = True
        break

with open("drift_status.txt", "w") as f:
    f.write("DRIFT_DETECTED" if drift_detected else "NO_DRIFT")