"""Independent label review.

  python scripts/label_review.py sheet   # writes data/review/review_sheet.csv (60 reports, labels hidden)
  python scripts/label_review.py score   # compares the filled sheet with the generator labels

The reviewer fills `reviewer_category` and `reviewer_urgency` for every row WITHOUT looking at
data/incidents.csv, using only the labelling guide in DATA.md. `score` writes
results/LABEL_REVIEW.md with agreement, Cohen's kappa and every disagreement.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data" / "review" / "review_sheet.csv"
N = 60


def make_sheet():
    df = pd.read_csv(ROOT / "data" / "incidents.csv")
    # Oversample hard reports: that is where label mistakes would hide.
    hard = df[df.difficulty == "hard"].sample(20, random_state=7)
    rest = df.drop(hard.index).groupby("category", group_keys=False).apply(
        lambda g: g.sample(min(len(g), (N - 20) // 6 + 1), random_state=7)).head(N - 20)
    sheet = pd.concat([hard, rest]).sample(frac=1, random_state=7)[["id", "text"]]
    sheet["reviewer_category"] = ""
    sheet["reviewer_urgency"] = ""
    sheet["reviewer_notes"] = ""
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    if SHEET.exists():
        sys.exit(f"{SHEET} already exists - refusing to overwrite a review in progress")
    sheet.to_csv(SHEET, index=False)
    print(f"wrote {len(sheet)} rows to {SHEET.relative_to(ROOT)}")


def score_sheet():
    sheet = pd.read_csv(SHEET).fillna("")
    filled = sheet[(sheet.reviewer_category != "") & (sheet.reviewer_urgency != "")]
    if len(filled) < 50:
        sys.exit(f"only {len(filled)} rows reviewed - the brief requires at least 50")
    gold = pd.read_csv(ROOT / "data" / "incidents.csv").set_index("id")
    m = filled.join(gold[["category", "urgency", "difficulty"]], on="id")
    m["reviewer_category"] = m.reviewer_category.str.strip()
    m["reviewer_urgency"] = m.reviewer_urgency.str.strip().str.lower()
    cat_agree = (m.category == m.reviewer_category).mean()
    urg_agree = (m.urgency == m.reviewer_urgency).mean()
    lines = ["# Independent label review", "",
             f"Reviewed: {len(m)} reports ({(m.difficulty == 'hard').sum()} hard). "
             "Reviewer labelled blind from the text and the DATA.md guide.", "",
             "| | agreement | Cohen's kappa |", "|---|---|---|",
             f"| category | {cat_agree:.1%} | {cohen_kappa_score(m.category, m.reviewer_category):.3f} |",
             f"| urgency | {urg_agree:.1%} | {cohen_kappa_score(m.urgency, m.reviewer_urgency):.3f} |", "",
             "## Disagreements", "", "| id | generator label | reviewer label | notes | report |", "|---|---|---|---|---|"]
    bad = m[(m.category != m.reviewer_category) | (m.urgency != m.reviewer_urgency)]
    for _, r in bad.iterrows():
        lines.append(f"| {r.id} | {r.category}/{r.urgency} | {r.reviewer_category}/{r.reviewer_urgency} | "
                     f"{r.reviewer_notes} | {r.text.replace('|', '/')} |")
    lines += ["", "## Resolution", "", "_Fill in: which disagreements were label errors (fix the template in "
              "data/generate.py) and which were genuinely ambiguous (document in DATA.md)._"]
    out = ROOT / "results" / "LABEL_REVIEW.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"category agreement {cat_agree:.1%}, urgency agreement {urg_agree:.1%}, "
          f"{len(bad)} disagreements -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    {"sheet": make_sheet, "score": score_sheet}[sys.argv[1] if len(sys.argv) > 1 else "sheet"]()
