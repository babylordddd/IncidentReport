"""Load data, fit the classical models and calibrators. Used by the scripts and the demo."""

import json
from pathlib import Path

import pandas as pd

from . import rules
from .llm import model_slug
from .ml import GRID, TfidfTriage
from .policy import PlattCalibrator

ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "results" / "llm"


def load_v1():
    return pd.read_csv(ROOT / "data" / "incidents.csv")


def load_v2():
    return pd.read_csv(ROOT / "data" / "incidents_v2.csv")


def fit_tfidf(train, val=None):
    """Fit TF-IDF + logreg; if `val` is given, pick features/C on val (never on test)."""
    def fit(features, C):
        return TfidfTriage(features, C).fit(train.text.tolist(), train.category.tolist(), train.urgency.tolist())

    if val is None:
        return fit("word", 1.0)
    best, best_score = None, -1.0
    for features, C in GRID:
        m = fit(features, C)
        preds = m.predict_many(val.text.tolist())
        acc = sum(p["category"] == c and p["urgency"] == u
                  for p, c, u in zip(preds, val.category, val.urgency)) / len(val)
        if acc > best_score:
            best, best_score = m, acc
    return best


def fit_calibrator(preds, df):
    ok = [(p["p_urgent"], int(u == "urgent")) for p, u in zip(preds, df.urgency) if not p.get("error")]
    return PlattCalibrator().fit([a for a, _ in ok], [b for _, b in ok])


def llm_cache(model, dataset="v1", taxonomy="v1"):
    """Cached LLM answers by id. `model` may be the real name or its file slug."""
    path = LLM_DIR / f"{model_slug(model)}__{dataset}__tax-{taxonomy}.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def cached_llm_models(dataset="v1", taxonomy="v1"):
    suffix = f"__{dataset}__tax-{taxonomy}.jsonl"
    return sorted(p.name[: -len(suffix)] for p in LLM_DIR.glob(f"*{suffix}"))


def rules_preds(df):
    return [rules.predict(t) for t in df.text]


def tfidf_preds(model, df):
    # Predict one at a time so latency reflects a single incoming report.
    return [model.predict(t) for t in df.text]
