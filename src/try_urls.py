import sys

import joblib
import pandas as pd

from src.features import FEATURE_NAMES, features_for

model = joblib.load("models/model.joblib")
urls = sys.argv[1:] or [
    "https://github.com/vidathhh9/hello-api",
    "https://www.github.com/vidathhh9/hello-api",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.bbc.com/news",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://google.com",
    "https://www.google.com",
    "http://paypal-secure-login.example-verify.xyz/account/update",
    "http://192.168.10.5/bank/login.php?id=88123",
    "https://www.paypa1.com",
]
rows = pd.DataFrame([features_for(u) for u in urls])[FEATURE_NAMES]
for u, p in zip(urls, model.predict_proba(rows)[:, 1]):
    print(f"{p:.2f}  {u}")
