import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = ("login", "verify", "secure", "account", "update", "bank", "confirm", "password")
IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "digit_ratio",
    "subdomain_count",
    "uses_https",
    "has_at_symbol",
    "host_is_ip",
    "suspicious_word_count",
    "special_char_count",
]


def extract(url: str) -> dict:
    url = url.strip().lower()
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.hostname or ""
        path = parsed.path or ""
        scheme = parsed.scheme
    except ValueError:
        host, path, scheme = "", "", ""
    digits = sum(c.isdigit() for c in url)
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": digits,
        "digit_ratio": digits / len(url) if url else 0.0,
        "subdomain_count": max(host.count(".") - 1, 0),
        "uses_https": int(scheme == "https"),
        "has_at_symbol": int("@" in url),
        "host_is_ip": int(bool(IP_PATTERN.match(host))),
        "suspicious_word_count": sum(w in url for w in SUSPICIOUS_WORDS),
        "special_char_count": sum(not c.isalnum() for c in url),
    }


def origin(url: str) -> str:
    url = url.strip().lower()
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        host = p.hostname or ""
        if host.startswith("www."):
            host = host[4:]
        return f"{p.scheme}://{host}"
    except ValueError:
        return ""


def features_for(url: str) -> dict:
    return extract(origin(url))
