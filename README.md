# Logistics Incident Triage

Classifies incoming incident reports for a logistics operator, routes urgent cases to an on-call
dispatcher, and sends the reports it is unsure about to a human. Routing decisions minimise the
**expected business cost**, so changing a penalty changes behaviour without retraining.

- **One-page business summary:** [BUSINESS_SUMMARY.md](BUSINESS_SUMMARY.md)
- **Evaluation results and error analysis:** [results/REPORT.md](results/REPORT.md)
- **Changed requirement:** [results/CHANGED_REQUIREMENT.md](results/CHANGED_REQUIREMENT.md)
- **Data provenance and labelling guide:** [DATA.md](DATA.md) · label review: [results/LABEL_REVIEW.md](results/LABEL_REVIEW.md)
- **Attribution (people and AI tools):** [ATTRIBUTION.md](ATTRIBUTION.md) · demo feedback: [FEEDBACK.md](FEEDBACK.md)

## Problem framing

| | |
|---|---|
| Input | free-text incident report (email, driver message, portal ticket) |
| Output | category (6 teams), urgency (`urgent` → page on-call within 1h), or **human review** |
| What matters most | not missing urgent cases (injury, hazmat, blocked site, theft in progress, perishable medical cargo) |
| Constraint | supervisors can review only ~25% of volume; LLM calls cost money and add latency |

Cost model (`triage/taxonomy.py`, illustrative figures):

| outcome | cost |
|---|---|
| urgent case routed as normal (**missed urgent**) | $500 |
| normal case paged to on-call (false alarm) | $40 |
| right urgency, wrong team | $25 |
| human review | $6 |

## Approaches compared

| # | Approach | Code |
|---|---|---|
| 0 | Keyword rules (baseline) | `triage/rules.py` |
| 1 | TF-IDF + logistic regression (hyper-parameters picked on val) | `triage/ml.py` |
| 2a | Zero-shot hosted LLM: Google `gemini-3.5-flash-lite` (free tier), JSON-schema output | `triage/llm.py` |
| 2b | Zero-shot local LLM: `qwen2.5:7b` via Ollama (no API, no data leaves the machine) | `triage/llm.py` |
| 3 | Hybrid cascade: TF-IDF first, LLM only when TF-IDF is unsure | `scripts/evaluate.py` |

Each approach can run with the shared **cost-aware policy** (`triage/policy.py`). Urgency probability is
calibrated on val (Platt scaling). For each report the policy compares the expected cost of
*route urgent*, *route normal* and *human review*, and spends the review budget where it saves the most.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/generate.py              # rebuild the dataset (deterministic)
python scripts/evaluate.py           # all approaches -> results/REPORT.md, metrics.json, figures
python scripts/changed_requirement.py
streamlit run app.py                 # demo at http://localhost:8501
```

No API key is needed to reproduce the results. The LLMs' raw answers are committed in `results/llm/*.jsonl`,
and the evaluation reads them from there. To re-run the LLMs yourself:

```bash
# Gemini: free API key from https://aistudio.google.com
export GEMINI_API_KEY=...
python scripts/run_llm.py --rpm 15                                  # val+test reports, free-tier pace
python scripts/run_llm.py --rpm 15 --dataset v2 --taxonomy v1       # old prompt on new data
python scripts/run_llm.py --rpm 15 --dataset v2 --taxonomy v2       # prompt with the new category

# Local: install Ollama (https://ollama.com), then
ollama pull qwen2.5:7b
python scripts/run_llm.py --model ollama/qwen2.5:7b --workers 1     # repeat with --dataset/--taxonomy as above
```

`claude-*` model names also work (set `ANTHROPIC_API_KEY`), as an optional paid comparison.

Delete a `results/llm/*.jsonl` file to force a fresh run. Otherwise the script resumes and only calls the API for missing ids.

## Repository layout

```
app.py                      Streamlit demo (triage one report, batch CSV, results, changed requirement)
data/generate.py            synthetic dataset generator (seed 2049)
data/incidents*.csv         generated data; data/review/ blind label-review sheet
triage/taxonomy.py          categories, routes, cost model  <- business rules live here
triage/{rules,ml,llm}.py    the three classifiers
triage/policy.py            calibration + cost-aware route / abstain decision
triage/evaluation.py        metrics
scripts/                    run_llm, evaluate, changed_requirement, label_review
results/                    generated reports, metrics, figures, error lists, cached LLM outputs
```
