import os
import re
import pandas as pd
import xgboost as xgb

TRAIN_PATH = "C:/Users/Dell/Desktop/data/train.csv"
VAL_PATH = "C:/Users/Dell/Desktop/data/val.csv"

DATA_DIR = os.path.dirname(TRAIN_PATH)
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "xgb_model.json")

print("Loading train and validation datasets...")
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

def clean_column_names(df):
    regex = re.compile(r"[\[\]<>]")
    df.columns = [regex.sub("_", col) for col in df.columns]
    return df

train_df = clean_column_names(train_df)
val_df = clean_column_names(val_df)

X_train = train_df.drop(columns=["score"])
y_train = train_df["score"]

X_val = val_df.drop(columns=["score"])
y_val = val_df["score"]

print("Training XGBoost Regressor model...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save_model(MODEL_SAVE_PATH)

print("\nModel Training Completed Successfully!")
print("Saved XGBoost model to:", MODEL_SAVE_PATH, "\n")