import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

def generate_report():

    ref_data = pd.read_csv("processed_data.csv")
    curr_data = pd.read_csv("test.csv")

    common_cols = [c for c in ref_data.columns if c in curr_data.columns]
    ref_data = ref_data[common_cols]
    curr_data = curr_data[common_cols]

    ref_ds = Dataset.from_pandas(ref_data, data_definition=DataDefinition())
    curr_ds = Dataset.from_pandas(curr_data, data_definition=DataDefinition())

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=curr_ds, reference_data=ref_ds)
    snapshot.save_html("drift_report.html")

if __name__ == "__main__":
    generate_report()