import pandas as pd

df = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
print(df.shape)
print(df.columns.tolist())
print(df["label"].value_counts())
print("duplicate URLs:", df["URL"].duplicated().sum())
print(df[["URL", "label"]].sample(10, random_state=1))
