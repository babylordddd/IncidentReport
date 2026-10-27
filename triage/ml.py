"""Approach 1 - TF-IDF + logistic regression. Two heads: category and urgency."""

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union


FEATURE_SETS = {
    "word": lambda: TfidfVectorizer(sublinear_tf=True, stop_words="english"),
    "word12": lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
    "word+char": lambda: make_union(
        TfidfVectorizer(sublinear_tf=True, stop_words="english"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    ),
}
GRID = [(f, C) for f in FEATURE_SETS for C in (0.3, 1.0, 10.0)]


class TfidfTriage:
    def __init__(self, features="word", C=1.0):
        self.config = {"features": features, "C": C}
        self.cat = make_pipeline(FEATURE_SETS[features](), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        self.urg = make_pipeline(FEATURE_SETS[features](), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))

    def fit(self, texts, categories, urgencies):
        self.cat.fit(texts, categories)
        self.urg.fit(texts, [u == "urgent" for u in urgencies])
        return self

    @property
    def classes(self):
        return list(self.cat.classes_)

    def predict_many(self, texts):
        t0 = time.perf_counter()
        cp = self.cat.predict_proba(texts)
        up = self.urg.predict_proba(texts)[:, 1]
        per_item_ms = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
        out = []
        for probs, pu in zip(cp, up):
            i = int(np.argmax(probs))
            out.append({
                "category": self.classes[i],
                "urgency": "urgent" if pu >= 0.5 else "normal",
                "p_urgent": float(pu),
                "category_confidence": float(probs[i]),
                "category_probs": {c: float(p) for c, p in zip(self.classes, probs)},
                "latency_ms": per_item_ms,
                "cost_usd": 0.0,
            })
        return out

    def predict(self, text):
        t0 = time.perf_counter()
        r = self.predict_many([text])[0]
        r["latency_ms"] = (time.perf_counter() - t0) * 1000
        return r
