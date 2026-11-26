import json

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/features.csv")
X = df.drop(columns="is_phishing")
y = df["is_phishing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

info = {
    "model": "RandomForestClassifier",
    "sklearn_version": sklearn.__version__,
    "feature_names": list(X.columns),
    "preprocessing": "features_for(url): scheme://host (leading www. removed), then extract()",
    "positive_class": "phishing",
    "threshold": 0.5,
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "dataset": "PhiUSIIL Phishing URL Dataset (UCI, CC BY 4.0)",
}

joblib.dump(model, "models/model.joblib")
with open("models/model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print(info)
