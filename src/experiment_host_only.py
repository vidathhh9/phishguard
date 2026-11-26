import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, extract


def origin(url):
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        return f"{p.scheme}://{p.hostname or ''}"
    except ValueError:
        return ""


raw = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv").drop_duplicates(subset="URL")
X = pd.DataFrame([extract(origin(u)) for u in raw["URL"]])[FEATURE_NAMES]
y = (raw["label"] == 0).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("precision", round(precision_score(y_test, pred), 3), "recall", round(recall_score(y_test, pred), 3))

tests = [
    "https://github.com/vidathhh9/hello-api",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.bbc.com/news/technology-68123456",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.google.com",
    "http://paypal-secure-login.example-verify.xyz/account/update",
    "http://192.168.10.5/bank/login.php?id=88123",
    "https://www.paypa1.com",
]
rows = pd.DataFrame([extract(origin(u)) for u in tests])[FEATURE_NAMES]
print()
print("Handmade test, host only (probability of phishing):")
for u, p in zip(tests, model.predict_proba(rows)[:, 1]):
    print(f"  {p:.2f}  {u}")
