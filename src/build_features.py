import pandas as pd

from src.features import FEATURE_NAMES, features_for, origin

df = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
df = df.assign(origin=df["URL"].map(origin)).drop_duplicates(subset="origin")

X = pd.DataFrame([features_for(u) for u in df["URL"]])[FEATURE_NAMES]
X["is_phishing"] = (df["label"] == 0).astype(int).values

X.to_csv("data/processed/features.csv", index=False)
print(X.shape)
print(X["is_phishing"].value_counts())
