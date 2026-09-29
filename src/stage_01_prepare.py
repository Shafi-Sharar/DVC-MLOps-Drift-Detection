import os
import pandas as pd

CSV_PATH = "C:/Users/Dell/Desktop/data/Anime.csv"

SAVE_PATH = os.path.join(os.path.dirname(CSV_PATH), "processed_data.csv")

print("Loading dataset for XGBoost setup...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("Preprocessing numerical values...")
df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(df["score"].median())
df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce").fillna(df["episodes"].median())
df["members"] = pd.to_numeric(df["members"], errors="coerce").fillna(0)
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(df["popularity"].max())

if "rank" in df.columns:
    df["ranked"] = pd.to_numeric(df["rank"], errors="coerce").fillna(df["rank"].max())
else:
    df["ranked"] = 0

df["type"] = df["type"].fillna("Unknown")
df["genre"] = df["genre"].fillna("")

print("Encoding categories...")
df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

genre_dummies = df["genre"].str.get_dummies(sep=", ")
genre_dummies.columns = [f"genre_{col.strip()}" for col in genre_dummies.columns]

processed_df = pd.concat([
    df[["episodes", "members", "popularity", "ranked"]],
    df[[col for col in df.columns if col.startswith("type_")]],
    genre_dummies,
    df["score"]], axis=1)

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
processed_df.to_csv(SAVE_PATH, index=False)

print("Setup Completed Successfully!")
print("Dataset Shape:", processed_df.shape)
print("Saved to:", SAVE_PATH, "\n")