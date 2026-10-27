"""Incident triage demo.  Run:  streamlit run app.py

Works without an API key (rules + TF-IDF live; LLM answers from the cached runs when the
report text is in the dataset). Set GEMINI_API_KEY, or run Ollama locally, to classify any text live.
"""

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

from triage import pipeline as P
from triage import rules
from triage.llm import DEFAULT_MODEL, LOCAL_MODEL, PRICES, LLMTriage
from triage.policy import decide, fit_review_margin
from triage.taxonomy import CATEGORIES_V1, CATEGORIES_V2, NEW_CATEGORY, CostModel

ROOT = Path(__file__).parent
st.set_page_config(page_title="Incident Triage", page_icon="🚚", layout="wide")


def llm_available(model):
    if model.startswith("gemini"):
        return bool(os.environ.get("GEMINI_API_KEY"))
    if model.startswith("claude"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:  # Ollama: is the local server up?
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except OSError:
        return False


@st.cache_resource
def models(with_new_category, llm):
    v1, v2 = P.load_v1(), P.load_v2()
    train, val = v1[v1.split == "train"], v1[v1.split == "val"]
    if with_new_category:
        train = pd.concat([train, v2[v2.category == NEW_CATEGORY].head(20)])
    ml = P.fit_tfidf(train, val)
    rules_kw = rules.KEYWORDS_V2 if with_new_category else rules.KEYWORDS
    rp_val = [rules.predict(t, rules_kw) for t in val.text]
    mp_val = P.tfidf_preds(ml, val)
    out = {"ml": ml, "rules_kw": rules_kw, "val": val,
           "val_preds": {"Rules": rp_val, "TF-IDF": mp_val},
           "cal": {"Rules": P.fit_calibrator(rp_val, val), "TF-IDF": P.fit_calibrator(mp_val, val)}}
    tax = "v2" if with_new_category else "v1"
    lp_cache = P.llm_cache(llm)
    if all(i in lp_cache for i in val.id):
        out["val_preds"]["LLM"] = [lp_cache[i] for i in val.id]
        out["cal"]["LLM"] = P.fit_calibrator(out["val_preds"]["LLM"], val)
    # Index cached LLM answers by text so the demo works offline on dataset reports.
    by_text = {}
    for dataset, df in (("v1", v1), ("v2", v2)):
        cache = P.llm_cache(llm, dataset, tax if dataset == "v2" else "v1")
        for row in df.itertuples():
            if row.id in cache:
                by_text[row.text] = cache[row.id]
    out["llm_by_text"] = by_text
    return out


@st.cache_resource
def llm_client(with_new_category, model):
    return LLMTriage(model=model, categories=CATEGORIES_V2 if with_new_category else CATEGORIES_V1)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Business rules")
missed = st.sidebar.slider("Cost of missing an urgent case ($)", 100, 5000, 500, 100)
false_urg = st.sidebar.slider("Cost of a false page to on-call ($)", 5, 200, 40, 5)
wrong_q = st.sidebar.slider("Cost of routing to the wrong team ($)", 5, 200, 25, 5)
review = st.sidebar.slider("Cost of a human review ($)", 1, 50, 6, 1)
capacity = st.sidebar.slider("Human review capacity (% of reports)", 0, 100, 25, 5) / 100
new_cat = st.sidebar.toggle(f"Enable new category `{NEW_CATEGORY}`", value=False,
                            help="Changed requirement: cold-chain contract signed.")
llm_options = [DEFAULT_MODEL, LOCAL_MODEL] + [m for m in PRICES if m not in (DEFAULT_MODEL, LOCAL_MODEL)]
model = st.sidebar.selectbox("LLM", llm_options)
has_key = llm_available(model)
st.sidebar.caption("Live LLM: " + ("on" if has_key else
                                   "off - cached answers only (set GEMINI_API_KEY / start Ollama)"))
costs = CostModel(missed_urgent=missed, false_urgent=false_urg, wrong_queue=wrong_q, human_review=review)
st.sidebar.caption(f"Page on-call when P(urgent) > {false_urg / (false_urg + missed):.1%}")

M = models(new_cat, model)

st.title("🚚 Logistics incident triage")
tab_triage, tab_batch, tab_eval, tab_change = st.tabs(["Triage a report", "Batch", "Evaluation", "Changed requirement"])


def run_all(text, live_llm=True):
    preds = {"Rules": rules.predict(text, M["rules_kw"]), "TF-IDF": M["ml"].predict(text)}
    if text in M["llm_by_text"]:
        preds["LLM"] = {**M["llm_by_text"][text], "_source": "cached"}
    elif has_key and live_llm:
        try:
            preds["LLM"] = {**llm_client(new_cat, model).predict(text), "_source": "live"}
        except Exception as e:  # show the failure, keep the other approaches usable
            preds["LLM"] = {"error": True, "category": None, "rationale": str(e), "_source": "error"}
    decisions = {}
    for name, p in preds.items():
        cal = M["cal"].get(name)
        val_preds = M["val_preds"].get(name)
        margin = fit_review_margin(val_preds, costs, cal, capacity) if val_preds else 0.0
        decisions[name] = decide(p, costs, cal, review_margin=margin)
    return preds, decisions


with tab_triage:
    test = P.load_v1().query("split == 'test'")
    examples = {"(type your own)": ""} | {f"{r.id} [{r.category}/{r.urgency}] {r.text[:70]}...": r.text
                                          for r in test.itertuples()}
    if new_cat:
        v2 = P.load_v2()
        examples |= {f"{r.id} [{r.category}/{r.urgency}] {r.text[:70]}...": r.text
                     for r in v2[v2.category == NEW_CATEGORY].tail(10).itertuples()}
    pick = st.selectbox("Example from the held-out test set (true label shown in brackets)", list(examples))
    text = st.text_area("Incident report", value=examples[pick], height=110,
                        placeholder="e.g. Forklift clipped the racking at Dock 3, one picker has a cut hand.")
    if st.button("Triage", type="primary") and text.strip():
        preds, decisions = run_all(text.strip())
        # Recommended system: TF-IDF first, the LLM when TF-IDF would send to review.
        final = decisions["TF-IDF"]
        source = "TF-IDF (confident)"
        if final["abstain"] and "LLM" in decisions:
            final, source = decisions["LLM"], f"{model} (TF-IDF unsure)"
        box = st.error if final["urgency"] == "urgent" and not final["abstain"] else (
            st.warning if final["abstain"] else st.success)
        box(f"**Route → {final['route']}**  \nDecided by: {source}")

        cols = st.columns(len(preds))
        for col, (name, p) in zip(cols, preds.items()):
            d = decisions[name]
            with col:
                title = model if name == "LLM" else name
                st.subheader(title + (f" ({p.get('_source')})" if "_source" in p else ""))
                if p.get("error"):
                    st.write("Model error / refusal → human review")
                    st.caption(p.get("rationale", ""))
                    continue
                st.metric("Category", p["category"], f"conf {p['category_confidence']:.2f}", delta_color="off")
                st.metric("P(urgent), calibrated", f"{d['p_urgent_calibrated']:.2f}")
                st.write(f"**Decision:** {'🧑 human review' if d['abstain'] else d['urgency']}")
                st.write(f"**Route:** {d['route']}")
                st.caption(f"latency {p['latency_ms']:.0f} ms · cost ${p.get('cost_usd', 0):.5f}")
                if p.get("rationale"):
                    st.caption("Why: " + p["rationale"])
                with st.expander("Expected cost of each action"):
                    st.json({k: round(v, 2) for k, v in d["expected_costs"].items()})
        if "LLM" not in preds:
            st.info("LLM column hidden: report not in the cached runs and the selected LLM is not available.")

with tab_batch:
    st.write("Upload a CSV with a `text` column (optionally `id`). Rules + TF-IDF run on every row; "
             "the LLM runs only where TF-IDF is unsure (and only if it is available).")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        out = []
        bar = st.progress(0.0)
        for i, t in enumerate(df.text.astype(str)):
            preds, decs = run_all(t, live_llm=False)
            d = decs["TF-IDF"]
            if d["abstain"] and has_key:
                p = llm_client(new_cat, model).predict(t)
                d = decide(p, costs, M["cal"].get("LLM"))
            out.append({"text": t, "category": d["category"], "urgency": d["urgency"],
                        "human_review": d["abstain"], "route": d["route"]})
            bar.progress((i + 1) / len(df))
        res = pd.DataFrame(out)
        st.dataframe(res, use_container_width=True)
        st.download_button("Download routed CSV", res.to_csv(index=False), "triaged.csv")

with tab_eval:
    mpath = ROOT / "results" / "metrics.json"
    if mpath.exists():
        data = json.loads(mpath.read_text())
        for scen, s in data["scenarios"].items():
            st.subheader(f"{scen} - test n={s['test_size']}")
            t = pd.DataFrame(s["approaches"]).T[[
                "category_accuracy", "urgent_recall_all", "abstain_rate", "missed_urgent_auto",
                "false_urgent_auto", "business_cost_per_report", "compute_cost_per_1k", "latency_p50_ms"]]
            st.dataframe(t.style.format("{:.3f}"), use_container_width=True)
            fig = ROOT / "results" / "figures" / f"coverage_vs_cost__{scen}.png"
            if fig.exists():
                st.image(str(fig), width=640)
        st.caption("Full report with error analysis: results/REPORT.md")
    else:
        st.write("Run `python scripts/evaluate.py` first.")

with tab_change:
    p = ROOT / "results" / "CHANGED_REQUIREMENT.md"
    st.markdown(p.read_text() if p.exists() else "Run `python scripts/changed_requirement.py` first.")
    st.info("Try it live: raise the missed-urgent cost in the sidebar, or enable the new category.")
