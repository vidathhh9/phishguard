# PhishGuard

A machine-learning phishing URL detector. Given a URL, it estimates the probability that the site is phishing, using only features computed from the URL text.

## Status

Model training works. The prediction API, experiment tracking, CI/CD and cloud deployment are coming next.

## Data

PhiUSIIL Phishing URL Dataset (UCI, CC BY 4.0): 134,850 legitimate and 100,945 phishing URLs. The dataset is not stored in this repo.

## How it works

1. `src/features.py` reduces a URL to `scheme://host` (leading `www.` removed) and computes 13 numeric features.
2. `src/build_features.py` builds the training table.
3. `src/train.py` trains a random forest and saves the model with a record of its features and metrics.

The same `features_for()` function will be used by the API, so training and prediction compute features identically.

## Results (test set)

| Version | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Full URL features | 0.999 | 0.991 | 0.995 |
| Host only, before removing www | 0.972 | 0.896 | 0.932 |
| Host only, www removed (current) | 0.978 | 0.885 | 0.930 |

The first score looks best but fails on real URLs: it flagged Wikipedia, BBC and Python docs as phishing, because in this dataset legitimate URLs almost never have a path.

## Limitations

- The dataset has shortcuts. Legitimate URLs are almost all bare homepages, all use https, and all started with `www`. I removed the path and `www` shortcuts. The https pattern remains.
- Look-alike domains such as `paypa1.com` are missed.
- The dataset is about 39% phishing. Real traffic is overwhelmingly legitimate, so false alarms would be more common in practice.
- Only URL text is used, so the model cannot see page content.
