import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, extract

df = pd.read_csv("data/processed/features.csv")
raw = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv").drop_duplicates(subset="URL")
urls = raw["URL"].reset_index(drop=True)

X = df.drop(columns="is_phishing")
y = df["is_phishing"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("Confusion matrix (rows = truth, columns = prediction):")
print(confusion_matrix(y_test, pred))

print()
for threshold in (0.3, 0.5, 0.7):
    p = (proba >= threshold).astype(int)
    missed = int(((p == 0) & (y_test == 1)).sum())
    false_alarms = int(((p == 1) & (y_test == 0)).sum())
    print(f"threshold {threshold}: missed phishing={missed}, false alarms={false_alarms}")

print()
print("Top features:")
for score, name in sorted(zip(model.feature_importances_.round(3), X.columns), reverse=True)[:6]:
    print(score, name)

print()
print("Phishing URLs the model missed:")
for i in y_test[(pred == 0) & (y_test == 1)].index[:10]:
    print(" ", urls.iloc[i])

print()
print("Legitimate URLs flagged as phishing:")
for i in y_test[(pred == 1) & (y_test == 0)].index[:10]:
    print(" ", urls.iloc[i])

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
rows = pd.DataFrame([extract(u) for u in tests])[FEATURE_NAMES]
print()
print("Handmade test (probability of phishing):")
for u, p in zip(tests, model.predict_proba(rows)[:, 1]):
    print(f"  {p:.2f}  {u}")
