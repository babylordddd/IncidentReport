"""Evaluate every approach on held-out test reports, under two split scenarios.

  python scripts/evaluate.py

Scenarios
  unseen_wording  test reports come from templates never seen in train (column `split`) - primary
  seen_wording    stratified random split (column `split_random`) - wording resembles history

Writes results/metrics.json, results/REPORT.md, results/errors/*.csv and results/figures/*.png.
LLM approaches are included for every model with a complete cached run in results/llm/.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triage import pipeline as P  # noqa: E402
from triage.evaluation import apply_policy, score  # noqa: E402
from triage.policy import expected_costs, fit_review_margin  # noqa: E402
from triage.taxonomy import DEFAULT_COSTS  # noqa: E402

RES = ROOT / "results"
REVIEW_CAPACITY = 0.25  # supervisors can review at most 25% of incoming reports
SCENARIOS = {"unseen_wording": "split", "seen_wording": "split_random"}


def policy(preds_val, val, preds_test, costs, capacity):
    cal = P.fit_calibrator(preds_val, val)
    margin = fit_review_margin(preds_val, costs, cal, capacity)
    return apply_policy(preds_test, costs, cal, review_margin=margin), cal, margin


def build_approaches(df, split_col, costs=DEFAULT_COSTS, capacity=REVIEW_CAPACITY):
    """Return {name: (preds, decisions)} on the test reports of one split scenario."""
    train, val, test = (df[df[split_col] == s] for s in ("train", "val", "test"))
    out, info = {}, {}

    rp_val, rp_test = P.rules_preds(val), P.rules_preds(test)
    out["rules (no abstain)"] = (rp_test, apply_policy(rp_test, costs, None, allow_abstain=False))
    out["rules + policy"] = (rp_test, policy(rp_val, val, rp_test, costs, capacity)[0])

    ml = P.fit_tfidf(train, val)
    info["tfidf_config"] = ml.config
    mp_val, mp_test = P.tfidf_preds(ml, val), P.tfidf_preds(ml, test)
    out["tfidf-logreg (argmax)"] = (mp_test, [_argmax(p) for p in mp_test])
    ml_dec, _, _ = policy(mp_val, val, mp_test, costs, capacity)
    out["tfidf-logreg + policy"] = (mp_test, ml_dec)

    for model in P.cached_llm_models():
        cache = P.llm_cache(model)
        if not all(i in cache for i in pd.concat([val.id, test.id])):
            print(f"skipping {model}: cache incomplete (run scripts/run_llm.py --model {model})")
            continue
        lp_val, lp_test = [cache[i] for i in val.id], [cache[i] for i in test.id]
        out[f"{model} (argmax)"] = (lp_test, [_argmax(p) for p in lp_test])
        llm_dec, _, _ = policy(lp_val, val, lp_test, costs, capacity)
        out[f"{model} + policy"] = (lp_test, llm_dec)

        # Cascade: free TF-IDF first; the LLM only sees the half of reports TF-IDF is least
        # sure about (escalation budget 50%), and the LLM's own policy decides those.
        ml_esc, _, _ = policy(mp_val, val, mp_test, costs, capacity=0.5)
        hp, hd = [], []
        for m_p, m_d, l_p, l_d in zip(mp_test, ml_esc, lp_test, llm_dec):
            if m_d["abstain"]:
                hp.append({**l_p, "latency_ms": m_p["latency_ms"] + l_p["latency_ms"]})
                hd.append(l_d)
            else:
                hp.append(m_p)
                hd.append(m_d)
        out[f"hybrid: tfidf -> {model}"] = (hp, hd)
    return out, info


def _argmax(p):
    if p.get("error") or p.get("category") is None:
        return {"category": None, "urgency": None, "abstain": True}
    return {"category": p["category"], "urgency": "urgent" if p["p_urgent"] >= 0.5 else "normal", "abstain": False}


def coverage_curve(preds, test, costs=DEFAULT_COSTS):
    """Business cost per report when abstaining on the least-confident share of reports."""
    rows = []
    for p, (_, r) in zip(preds, test.iterrows()):
        if p.get("error"):
            continue
        ec = expected_costs(p["p_urgent"], p["category_confidence"], costs)
        urg = "urgent" if ec["route_urgent"] <= ec["route_normal"] else "normal"
        risk = min(ec["route_urgent"], ec["route_normal"])
        rows.append((risk, costs.outcome_cost(r.category, r.urgency, p["category"], urg, False)))
    rows.sort(key=lambda x: x[0])
    pts = []
    for k in range(len(rows) + 1):
        total = sum(c for _, c in rows[:k]) + (len(rows) - k) * costs.human_review
        pts.append((k / len(rows), total / len(rows)))
    return pts


def run_scenario(df, scenario, split_col):
    test = df[df[split_col] == "test"]
    approaches, info = build_approaches(df, split_col)
    metrics, frames = {}, {}
    for name, (preds, decs) in approaches.items():
        m, rows = score(test, preds, decs)
        metrics[name] = m
        frames[name] = rows
        bad = rows[(~rows.abstain) & ((rows.true_category != rows.pred_category) |
                                      (rows.true_urgency != rows.pred_urgency))]
        bad.to_csv(RES / "errors" / f"{scenario}__{_slug(name)}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (preds, _) in approaches.items():
        if "+ policy" in name:
            xs, ys = zip(*coverage_curve(preds, test))
            ax.plot(xs, ys, label=name.replace(" + policy", ""))
    ax.axvline(1 - REVIEW_CAPACITY, ls="--", c="grey", lw=1)
    ax.text(1 - REVIEW_CAPACITY, ax.get_ylim()[1] * 0.95, " review capacity", color="grey", fontsize=8, va="top")
    ax.set_xlabel("Share of reports routed automatically (rest go to human review)")
    ax.set_ylabel("Business cost per report (USD)")
    ax.set_title(f"Cost vs automation coverage - {scenario} test split")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "figures" / f"coverage_vs_cost__{scenario}.png", dpi=150)
    plt.close(fig)
    return metrics, frames, info, len(test)


def main():
    df = P.load_v1()
    (RES / "errors").mkdir(parents=True, exist_ok=True)
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    all_metrics, report = {}, []
    for scenario, col in SCENARIOS.items():
        metrics, frames, info, n = run_scenario(df, scenario, col)
        all_metrics[scenario] = {"test_size": n, **info, "approaches": metrics}
        report.append(section(scenario, metrics, frames, info, n))
        print(f"\n== {scenario} (n={n}, tfidf {info['tfidf_config']})")
        print_table(metrics)

    with open(RES / "metrics.json", "w") as f:
        json.dump({"cost_model": DEFAULT_COSTS.__dict__, "review_capacity": REVIEW_CAPACITY,
                   "scenarios": all_metrics}, f, indent=2)
    header = ["# Evaluation report", "",
              "Generated by `scripts/evaluate.py` - do not edit by hand.", "",
              f"Cost model (USD per report): `{DEFAULT_COSTS.__dict__}`. Human review capacity: "
              f"{REVIEW_CAPACITY:.0%} of volume.", "",
              "- `(argmax)` / `(no abstain)`: every report is auto-routed on the model's top prediction.",
              "- `+ policy`: cost-aware routing (`triage/policy.py`). Urgency probability is calibrated on the "
              "val split; reports are sent to human review where that saves the most expected cost, "
              "up to the review capacity.",
              "- `hybrid`: TF-IDF handles reports it is confident about; the rest go to the LLM.",
              "- `business_cost_per_report` counts misroutes and reviews only; `total_cost_per_1k` adds LLM API cost.", ""]
    (RES / "REPORT.md").write_text("\n".join(header + report))


def section(scenario, metrics, frames, info, n):
    t = pd.DataFrame(metrics).T
    cols = ["category_accuracy", "category_macro_f1", "urgent_recall_all", "abstain_rate",
            "auto_fully_correct", "missed_urgent_auto", "false_urgent_auto",
            "business_cost_per_report", "compute_cost_per_1k", "total_cost_per_1k",
            "latency_p50_ms", "latency_p95_ms", "accuracy_easy", "accuracy_hard"]
    lines = [f"## Scenario: {scenario} (test n={n}, urgent={int(t.urgent_cases.iloc[0])})", "",
             f"TF-IDF configuration chosen on val: `{info['tfidf_config']}`", "",
             _md_table(t[cols]), "",
             f"![coverage](figures/coverage_vs_cost__{scenario}.png)", "",
             "### Auto-routed mistakes", ""]
    for name, rows in frames.items():
        bad = rows[(~rows.abstain) & ((rows.true_category != rows.pred_category) |
                                      (rows.true_urgency != rows.pred_urgency))]
        lines.append(f"**{name}** - {len(bad)} auto-routed errors, {int(rows.abstain.sum())} sent to review "
                     f"(full list: `results/errors/{scenario}__{_slug(name)}.csv`)")
        if len(bad):
            lines += ["", "| id | difficulty | true | predicted | report |", "|---|---|---|---|---|"]
            for _, r in bad.head(8).iterrows():
                lines.append(f"| {r.id} | {r.difficulty} | {r.true_category}/{r.true_urgency} | "
                             f"{r.pred_category}/{r.pred_urgency} | {r.text[:110].replace('|', '/')} |")
        lines.append("")
    return "\n".join(lines)


def print_table(metrics):
    cols = ["category_accuracy", "urgent_recall_all", "abstain_rate", "missed_urgent_auto",
            "business_cost_per_report", "compute_cost_per_1k", "latency_p50_ms"]
    t = pd.DataFrame(metrics).T[cols]
    t.columns = ["cat_acc", "urg_recall", "abstain", "missed_urg", "biz$/rep", "api$/1k", "p50_ms"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(t)


def _slug(name):
    for a, b in [(" ", "_"), (":", ""), ("->", "to"), ("(", ""), (")", ""), ("+", "plus")]:
        name = name.replace(a, b)
    return name


def _md_table(t):
    head = "| approach | " + " | ".join(t.columns) + " |"
    sep = "|---" * (len(t.columns) + 1) + "|"
    body = []
    for name, row in t.iterrows():
        cells = ["-" if v is None or (isinstance(v, float) and pd.isna(v)) else
                 f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        body.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


if __name__ == "__main__":
    main()
