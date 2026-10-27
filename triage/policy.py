"""Cost-aware routing policy shared by every approach.

Given a model's p(urgent) and category confidence, pick the action with the lowest
expected cost: auto-route as urgent, auto-route as normal, or send to human review.
Because the decision is driven by the CostModel, changing a penalty changes behaviour
without retraining anything.
"""

import math

from .taxonomy import DEFAULT_COSTS, route_for


class PlattCalibrator:
    """1-D logistic calibration of a probability, fitted on the validation split."""

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    @staticmethod
    def _logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    def fit(self, probs, labels, iters=3000, lr=0.05):
        xs = [self._logit(p) for p in probs]
        a, b = 1.0, 0.0
        for _ in range(iters):
            ga = gb = 0.0
            for x, y in zip(xs, labels):
                q = 1 / (1 + math.exp(-(a * x + b)))
                ga += (q - y) * x
                gb += q - y
            a -= lr * ga / len(xs)
            b -= lr * gb / len(xs)
        self.a, self.b = a, b
        return self

    def __call__(self, p):
        return 1 / (1 + math.exp(-(self.a * self._logit(p) + self.b)))


def expected_costs(p_urgent, cat_conf, costs=DEFAULT_COSTS):
    """Expected cost of each action. Category errors cost wrong_queue regardless of urgency."""
    wrong_cat = (1 - cat_conf) * costs.wrong_queue
    return {
        "route_urgent": (1 - p_urgent) * costs.false_urgent + wrong_cat,
        "route_normal": p_urgent * costs.missed_urgent + wrong_cat,
        "human_review": costs.human_review,
    }


def review_saving(pred, costs=DEFAULT_COSTS, calibrator=None):
    """Expected USD saved by sending this report to a human instead of auto-routing it."""
    p = calibrator(pred["p_urgent"]) if calibrator else pred["p_urgent"]
    ec = expected_costs(p, pred["category_confidence"], costs)
    return min(ec["route_urgent"], ec["route_normal"]) - ec["human_review"]


def fit_review_margin(preds, costs=DEFAULT_COSTS, calibrator=None, capacity=0.25):
    """Smallest saving threshold that keeps the review rate at or below `capacity` (fitted on val).

    Supervisors can only read so many reports, so we review the cases where review saves
    the most, up to the capacity. Returns 0 when capacity is not binding.
    """
    savings = sorted((review_saving(p, costs, calibrator) for p in preds if not p.get("error")), reverse=True)
    k = int(capacity * len(savings))
    if k >= len(savings) or savings[k] <= 0:
        return 0.0
    return savings[k]


def decide(pred, costs=DEFAULT_COSTS, calibrator=None, allow_abstain=True, review_margin=0.0):
    """Return the final decision dict for one prediction."""
    if pred.get("error") or pred.get("category") is None:
        return {"category": None, "urgency": None, "abstain": True,
                "route": route_for(None, None, True), "reason": "model error/refusal"}
    p = calibrator(pred["p_urgent"]) if calibrator else pred["p_urgent"]
    ec = expected_costs(p, pred["category_confidence"], costs)
    best_route = min(("route_urgent", "route_normal"), key=ec.get)
    abstain = allow_abstain and ec[best_route] - ec["human_review"] > review_margin
    action = "human_review" if abstain else best_route
    urgency = "urgent" if action == "route_urgent" else "normal"
    if abstain:
        # Supervisor still sees the model's best guess.
        urgency = "urgent" if p >= 0.5 else "normal"
    return {
        "category": pred["category"],
        "urgency": urgency,
        "abstain": abstain,
        "p_urgent_calibrated": p,
        "expected_costs": ec,
        "route": route_for(pred["category"], urgency, abstain),
        "reason": action,
    }
