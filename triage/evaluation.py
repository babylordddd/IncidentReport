"""Metrics shared by scripts/evaluate.py and scripts/changed_requirement.py."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .policy import decide
from .taxonomy import DEFAULT_COSTS


def apply_policy(preds, costs=DEFAULT_COSTS, calibrator=None, allow_abstain=True, review_margin=0.0):
    return [decide(p, costs, calibrator, allow_abstain, review_margin) for p in preds]


def score(df, preds, decisions, costs=DEFAULT_COSTS):
    """Return (metrics dict, per-row DataFrame) for one approach on one dataset."""
    rows = []
    for (_, r), p, d in zip(df.iterrows(), preds, decisions):
        biz = costs.outcome_cost(r.category, r.urgency, d["category"], d["urgency"], d["abstain"])
        rows.append({
            "id": r.id, "text": r.text, "difficulty": r.difficulty,
            "true_category": r.category, "true_urgency": r.urgency,
            "pred_category": d["category"], "pred_urgency": d["urgency"], "abstain": d["abstain"],
            "p_urgent": p.get("p_urgent"), "category_confidence": p.get("category_confidence"),
            "rationale": p.get("rationale", ""),
            "business_cost": biz, "compute_cost": p.get("cost_usd", 0.0), "latency_ms": p.get("latency_ms", 0.0),
        })
    out = pd.DataFrame(rows)
    auto = out[~out.abstain]
    urgent = out.true_urgency == "urgent"
    auto_urgent = auto.true_urgency == "urgent"
    missed = int(((auto.true_urgency == "urgent") & (auto.pred_urgency != "urgent")).sum())
    false_urg = int(((auto.true_urgency != "urgent") & (auto.pred_urgency == "urgent")).sum())
    cat_ok = out.true_category == out.pred_category
    n = len(out)
    m = {
        "n": n,
        # Model quality on every report (abstained ones scored on the model's best guess)
        "category_accuracy": float(cat_ok.mean()),
        "category_macro_f1": float(f1_score(out.true_category, out.pred_category.fillna("none"), average="macro")),
        "urgent_recall_all": float((out[urgent].pred_urgency == "urgent").mean()),
        "urgent_precision_all": _safe((out.pred_urgency == "urgent") & urgent, out.pred_urgency == "urgent"),
        # Outcomes after the routing policy (what the business experiences)
        "abstain_rate": float(out.abstain.mean()),
        "auto_routed": int(len(auto)),
        "auto_fully_correct": float(((auto.true_category == auto.pred_category) &
                                      (auto.true_urgency == auto.pred_urgency)).mean()) if len(auto) else None,
        "missed_urgent_auto": missed,
        "false_urgent_auto": false_urg,
        "urgent_cases": int(urgent.sum()),
        "urgent_auto_recall": float((auto[auto_urgent].pred_urgency == "urgent").mean()) if auto_urgent.any() else None,
        "business_cost_total": float(out.business_cost.sum()),
        "business_cost_per_report": float(out.business_cost.mean()),
        "compute_cost_per_1k": float(out.compute_cost.mean() * 1000),
        "total_cost_per_1k": float((out.business_cost + out.compute_cost).mean() * 1000),
        "latency_p50_ms": float(np.percentile(out.latency_ms, 50)),
        "latency_p95_ms": float(np.percentile(out.latency_ms, 95)),
        "accuracy_easy": float(cat_ok[out.difficulty == "easy"].mean()),
        "accuracy_hard": float(cat_ok[out.difficulty == "hard"].mean()),
    }
    return m, out


def _safe(num_mask, den_mask):
    d = den_mask.sum()
    return float(num_mask.sum() / d) if d else None
