import pandas as pd
from river import drift
from evidently import Report
from evidently.presets import DataDriftPreset

reference_data = pd.read_csv("train.csv")
current_data = pd.read_csv("test.csv")

adwin = drift.ADWIN()
drift_detected = False

for val in current_data.iloc[:, 0]: 
    adwin.update(val)
    if adwin.drift_detected:
        drift_detected = True
        break

with open("drift_status.txt", "w") as f:
    f.write("DRIFT_DETECTED" if drift_detected else "NO_DRIFT")

report = Report(metrics=[DataDriftPreset()])
eval_result=report.run(reference_data=reference_data, current_data=current_data)
eval_result.save_html("drift_report.html")


print("Visual report saved as 'drift_report.html'")
print(f"Overall Drift Status: {'DRIFT_DETECTED' if drift_detected else 'NO_DRIFT'}")
