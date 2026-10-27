"""Incident taxonomy, routing table and business cost model.

Everything that encodes a business decision lives here so a changed
requirement is a one-file edit (see scripts/changed_requirement.py).
"""

from dataclasses import dataclass, field

CATEGORIES_V1 = [
    "delivery_delay",
    "damaged_goods",
    "lost_or_stolen",
    "safety_incident",
    "vehicle_equipment_failure",
    "documentation_customs",
]

# Added in the "changed requirement" scenario (cold-chain contract signed).
NEW_CATEGORY = "temperature_excursion"
CATEGORIES_V2 = CATEGORIES_V1 + [NEW_CATEGORY]

URGENCIES = ["urgent", "normal"]

CATEGORY_DESCRIPTIONS = {
    "delivery_delay": "Shipment late, missed slot, stuck in traffic/queue, rescheduling. No damage or loss.",
    "damaged_goods": "Cargo or packaging physically damaged, crushed, wet, broken. No injury and no hazardous release.",
    "lost_or_stolen": "Shipment missing, mis-delivered, not scanned, theft, break-in, seal tampering.",
    "safety_incident": "Any injury, near miss, fire, spill/leak of hazardous material, unsafe site condition. Takes precedence over other categories.",
    "vehicle_equipment_failure": "Truck, trailer, forklift, reefer unit, scanner or conveyor breakdown with no injury.",
    "documentation_customs": "Missing/incorrect paperwork, customs hold, invoice or label mismatch, permits.",
    NEW_CATEGORY: "Temperature-controlled cargo outside its allowed range (cold chain breach), reefer alarm with product at risk.",
}

URGENCY_DEFINITION = (
    "urgent = needs action within 1 hour: a person is hurt or in danger, hazardous material released, "
    "a road/dock/site is blocked, theft in progress, or perishable/medical cargo will be lost without "
    "immediate action. Everything else is normal, even if the reporter is angry or writes in capitals."
)

ROUTES = {
    "delivery_delay": "Customer Service",
    "damaged_goods": "Claims",
    "lost_or_stolen": "Security & Claims",
    "safety_incident": "HSE (Health & Safety)",
    "vehicle_equipment_failure": "Fleet Maintenance",
    "documentation_customs": "Customs Desk",
    NEW_CATEGORY: "Cold Chain Desk",
}
URGENT_ROUTE = "On-call Dispatcher (paged)"
HUMAN_REVIEW_ROUTE = "Triage Supervisor (human review)"


@dataclass
class CostModel:
    """Expected cost (USD) of each outcome for one report.

    Figures are illustrative assumptions agreed for this exercise, not client data.
    They are documented in BUSINESS_SUMMARY.md and can be changed here.
    """

    missed_urgent: float = 500.0      # urgent case sent to a normal queue (SLA breach, escalation, risk)
    false_urgent: float = 40.0        # on-call dispatcher paged for a normal case
    wrong_queue: float = 25.0         # right urgency, wrong team: re-handoff, ~1 day delay
    human_review: float = 6.0         # supervisor reads and routes the report (~4 min)
    correct: float = 0.0
    extra: dict = field(default_factory=dict)

    def outcome_cost(self, true_cat, true_urg, pred_cat, pred_urg, abstained):
        if abstained:
            return self.human_review
        if true_urg == "urgent" and pred_urg != "urgent":
            return self.missed_urgent
        cost = 0.0
        if true_urg != "urgent" and pred_urg == "urgent":
            cost += self.false_urgent
        if true_cat != pred_cat:
            cost += self.wrong_queue
        return cost


DEFAULT_COSTS = CostModel()
STRICT_COSTS = CostModel(missed_urgent=2000.0)


def route_for(category, urgency, abstained=False):
    if abstained:
        return HUMAN_REVIEW_ROUTE
    team = ROUTES.get(category, HUMAN_REVIEW_ROUTE)
    return f"{URGENT_ROUTE} -> {team}" if urgency == "urgent" else team
