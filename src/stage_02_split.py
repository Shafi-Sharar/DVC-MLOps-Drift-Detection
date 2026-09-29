import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = "C:/Users/Dell/Desktop/data/processed_data.csv"

DATA_DIR = os.path.dirname(PROCESSED_DATA_PATH)

print("Loading processed data for splitting...")
df = pd.read_csv(PROCESSED_DATA_PATH)

X = df.drop(columns=["score"])
y = df["score"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

print("Data Splitting Completed Successfully!")
print("Train set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)
print("Test set shape:", test_df.shape, "\n")