"""Run the LLM classifier and cache raw predictions to results/llm/*.jsonl.

Gemini needs GEMINI_API_KEY; ollama/* needs a running Ollama server. Re-running skips ids already in the cache, so an
interrupted run resumes and graders can evaluate from the committed cache with no key.

  python scripts/run_llm.py                                  # v1 val+test (both split schemes), Gemini
  python scripts/run_llm.py --model ollama/qwen2.5:7b --workers 1
  python scripts/run_llm.py --dataset v2 --taxonomy v2       # changed-requirement run
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triage.llm import DEFAULT_MODEL, LLMTriage, model_slug  # noqa: E402
from triage.taxonomy import CATEGORIES_V1, CATEGORIES_V2  # noqa: E402


def cache_path(model, dataset, taxonomy):
    return ROOT / "results" / "llm" / f"{model_slug(model)}__{dataset}__tax-{taxonomy}.jsonl"


def load_cache(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dataset", choices=["v1", "v2"], default="v1")
    ap.add_argument("--taxonomy", choices=["v1", "v2"], default="v1")
    ap.add_argument("--splits", default="val,test",
                    help="v1 splits to run; a report is included if it is in these splits under either split scheme")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=float, default=0, help="max requests per minute (free-tier limits), 0 = no limit")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.dataset == "v1":
        df = pd.read_csv(ROOT / "data" / "incidents.csv")
        want = args.splits.split(",")
        df = df[df.split.isin(want) | df.split_random.isin(want)]
    else:
        df = pd.read_csv(ROOT / "data" / "incidents_v2.csv")
    if args.limit:
        df = df.head(args.limit)

    path = cache_path(args.model, args.dataset, args.taxonomy)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(path)
    todo = df[~df.id.isin(done)]
    print(f"{len(done)} cached, {len(todo)} to run -> {path.relative_to(ROOT)}")

    clf = LLMTriage(model=args.model, categories=CATEGORIES_V2 if args.taxonomy == "v2" else CATEGORIES_V1)

    lock, last = threading.Lock(), [0.0]

    def work(row):
        if args.rpm:
            with lock:  # space requests evenly to stay under the per-minute limit
                wait = last[0] + 60 / args.rpm - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.monotonic()
        try:
            return {"id": row.id, **clf.predict(row.text)}
        except Exception as e:  # keep going; failed ids are retried on the next run
            print(f"  {row.id}: {type(e).__name__}: {e}")
            return None

    n_ok = 0
    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(work, todo.itertuples()):
            if r:
                f.write(json.dumps(r) + "\n")
                f.flush()
                n_ok += 1
    total = load_cache(path)
    cost = sum(r["cost_usd"] for r in total.values())
    print(f"done: {n_ok} new, {len(total)} total, cumulative cost ${cost:.4f}")


if __name__ == "__main__":
    main()
