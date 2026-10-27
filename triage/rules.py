"""Approach 0 - keyword rules. The simplest thing an ops team could write in an afternoon."""

import re
import time

KEYWORDS = {
    "safety_incident": ["injur", "ambulance", "fire", "leak", "spill", "evacuat", "near miss", "collapsed",
                        "bleeding", "fell", "hi-vis", "safety", "trapped", "dizzy", "smoking", "first aid", "gas"],
    "lost_or_stolen": ["missing", "stolen", "theft", "break-in", "not scanned", "lost", "wrong pallet",
                       "proof of delivery", "seal", "tracker", "taken", "drove off", "wrong house", "cutting"],
    "damaged_goods": ["damage", "crushed", "broken", "smashed", "dented", "wet", "scratched", "torn",
                      "punctured", "collapsed", "fell off", "dropped"],
    "vehicle_equipment_failure": ["flat tyre", "broke down", "breakdown", "conveyor", "scanner", "warning light",
                                  "tail lift", "jamming", "wont start", "won't start", "sorter", "reefer",
                                  "fault", "engine", "brakes", "pallet jack", "leveller"],
    "documentation_customs": ["customs", "invoice", "paperwork", "certificate", "eori", "declaration",
                              "packing list", "label", "adr", "document", "licence", "border"],
    "delivery_delay": ["late", "delay", "eta", "slot", "waiting", "queue", "postpone", "reschedul",
                       "behind", "traffic", "hasn't come", "held up"],
}
# Changed requirement: the one edit needed to support the new cold-chain category.
KEYWORDS_V2 = {**KEYWORDS, "temperature_excursion": ["°c", "degrees", "temp", "chiller", "cold room", "logger",
                                                     "excursion", "setpoint", "cold-chain", "cold chain"]}

URGENT_WORDS = ["urgent", "asap", "now", "immediately", "ambulance", "fire", "injur", "trapped", "evacuat",
                "blocked", "can't get", "spoils", "right now", "emergency", "bleeding", "collapsed"]


def predict(text, keywords=KEYWORDS):
    t0 = time.perf_counter()
    low = text.lower()
    scores = {c: sum(low.count(k) for k in kws) for c, kws in keywords.items()}
    best = max(scores, key=scores.get)
    hits = scores[best]
    if hits == 0:
        best = "delivery_delay"  # most common queue as fallback
    urgent = any(re.search(r"\b" + re.escape(w), low) for w in URGENT_WORDS)
    # Rules have no real probabilities; use fixed confidence levels so the shared
    # cost policy can still treat "no keyword matched" as low confidence.
    cat_conf = 0.0 if hits == 0 else 0.5 if hits == 1 else 0.8
    return {
        "category": best,
        "urgency": "urgent" if urgent else "normal",
        "p_urgent": 0.85 if urgent else 0.15,
        "category_confidence": cat_conf,
        "category_probs": {best: cat_conf},
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "cost_usd": 0.0,
    }
