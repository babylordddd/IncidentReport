"""Changed requirement experiments.

  python scripts/changed_requirement.py

A. Stricter penalty: missing an urgent case now costs $2,000 instead of $500.
   Nothing is retrained - only the CostModel changes and the policy re-derives its decisions.
B. New category: the client signs a cold-chain contract and adds `temperature_excursion`.
   Uses data/incidents_v2.csv. The first 20 temperature reports are the "newly labelled
   examples" the team can produce in a day; everything else is the evaluation set.

Writes results/changed_requirement.json and results/CHANGED_REQUIREMENT.md.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import REVIEW_CAPACITY, _md_table, build_approaches  # noqa: E402

from triage import pipeline as P  # noqa: E402
from triage import rules  # noqa: E402
from triage.evaluation import apply_policy, score  # noqa: E402
from triage.policy import fit_review_margin  # noqa: E402
from triage.taxonomy import DEFAULT_COSTS, NEW_CATEGORY, STRICT_COSTS  # noqa: E402

RES = ROOT / "results"
N_NEW_LABELS = 20


def part_a(df):
    test = df[df.split == "test"]
    rows = {}
    # Both policies are scored under the NEW cost model: what does keeping the old policy cost?
    for label, costs in [("old policy (tuned for $500)", DEFAULT_COSTS), ("re-derived policy ($2000)", STRICT_COSTS)]:
        approaches, _ = build_approaches(df, "split", costs)
        for name, (preds, decs) in approaches.items():
            if "+ policy" not in name and "hybrid" not in name:
                continue
            m, _ = score(test, preds, decs, STRICT_COSTS)
            rows[(name, label)] = {
                "urgent_threshold_p": costs.false_urgent / (costs.false_urgent + costs.missed_urgent),
                "missed_urgent": m["missed_urgent_auto"], "false_urgent": m["false_urgent_auto"],
                "abstain_rate": m["abstain_rate"], "urgent_recall_after_policy": m["urgent_auto_recall"],
                "business_cost_per_report_at_2000": m["business_cost_per_report"],
            }
    return rows


def part_b(v1, v2):
    temp = v2[v2.category == NEW_CATEGORY]
    new_labels = temp.head(N_NEW_LABELS)
    ev = v2.drop(new_labels.index)
    is_new = (ev.category == NEW_CATEGORY).to_numpy()
    results = {}

    def record(name, preds, note, calibration=None):
        cal_preds, cal_df = calibration if calibration else (preds, ev)
        cal = P.fit_calibrator(cal_preds, cal_df)
        margin = fit_review_margin(cal_preds, DEFAULT_COSTS, cal, REVIEW_CAPACITY)
        decs = apply_policy(preds, DEFAULT_COSTS, cal, review_margin=margin)
        m, rows = score(ev, preds, decs)
        hit = rows.true_category == rows.pred_category
        results[name] = {
            "note": note,
            "category_accuracy": m["category_accuracy"],
            "new_category_recall": float(hit[is_new].mean()),
            "old_categories_accuracy": float(hit[~is_new].mean()),
            "urgent_recall": m["urgent_recall_all"],
            "abstain_rate": m["abstain_rate"],
            "business_cost_per_report": m["business_cost_per_report"],
            "labelled_examples_needed": note.split("|")[0].strip(),
        }

    v1_fit = v1[v1.split != "test"]
    v1_val = v1[v1.split == "val"]
    v1_val_rules = [rules.predict(t) for t in v1_val.text]
    record("rules v1", [rules.predict(t) for t in ev.text], "0 | no category for cold chain",
           (v1_val_rules, v1_val))
    record("rules v2 (+10 keywords)", [rules.predict(t, rules.KEYWORDS_V2) for t in ev.text],
           "0 | one keyword list added", (v1_val_rules, v1_val))

    ml_v1 = P.fit_tfidf(v1_fit)
    record("tfidf v1", P.tfidf_preds(ml_v1, ev), "0 | old model, cannot output the new class",
           (P.tfidf_preds(ml_v1, v1_val), v1_val))
    ml_v2 = P.fit_tfidf(pd.concat([v1_fit, new_labels]))
    record(f"tfidf retrained (+{N_NEW_LABELS} labels)", P.tfidf_preds(ml_v2, ev),
           f"{N_NEW_LABELS} | retrained with newly labelled reports", (P.tfidf_preds(ml_v2, v1_val), v1_val))

    for model in P.cached_llm_models("v2", "v1"):
        cache = P.llm_cache(model, "v2", "v1")
        if all(i in cache for i in ev.id):
            record(f"{model} v1 prompt", [cache[i] for i in ev.id], "0 | old prompt")
    for model in P.cached_llm_models("v2", "v2"):
        cache = P.llm_cache(model, "v2", "v2")
        if all(i in cache for i in ev.id):
            record(f"{model} v2 prompt", [cache[i] for i in ev.id], "0 | one line added to the prompt")
    return results, len(ev), int(is_new.sum())


def main():
    v1, v2 = P.load_v1(), P.load_v2()
    a = part_a(v1)
    b, n_ev, n_new = part_b(v1, v2)

    ta = pd.DataFrame(a).T
    ta.index = [f"{n} @ {c}" for n, c in ta.index]
    tb = pd.DataFrame(b).T.drop(columns=["note"])
    print(ta.to_string(float_format="{:.3f}".format))
    print(tb.to_string(float_format="{:.3f}".format))

    with open(RES / "changed_requirement.json", "w") as f:
        json.dump({"stricter_penalty": {k: v for k, v in zip(ta.index, ta.to_dict("records"))},
                   "new_category": b}, f, indent=2)
    md = ["# Changed requirement", "", "Generated by `scripts/changed_requirement.py`.", "",
          "## A. Missing an urgent case now costs $2,000 (was $500)", "",
          "No model is retrained. The policy's break-even probability for paging the on-call dispatcher "
          "falls from 7.4% to 2.0%, and the review budget is re-spent on the riskiest reports.", "",
          _md_table(ta), "",
          f"## B. New category `{NEW_CATEGORY}` (cold-chain contract)", "",
          f"Evaluation set: {n_ev} reports from `data/incidents_v2.csv` ({n_new} temperature excursions, "
          f"the rest old categories with partly new wording). The first {N_NEW_LABELS} temperature "
          "reports are held back as the labelled examples a team could produce in a day.", "",
          _md_table(tb), ""]
    (RES / "CHANGED_REQUIREMENT.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
