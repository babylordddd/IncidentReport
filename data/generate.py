"""Generate the synthetic, non-confidential incident report dataset.

Deterministic: `python data/generate.py` always writes the same files (seed 2049).

Outputs
  data/incidents.csv      360 reports, 6 categories (v1 taxonomy), with train/val/test split
  data/incidents_v2.csv    80 reports for the changed-requirement scenario
                           (40 temperature_excursion + 40 v1 reports from new templates)

Every report is built from a hand-written template whose label was assigned when the
template was written (see DATA.md for the labelling guide). A share of templates are
deliberately hard: angry-but-minor reports, calm-but-urgent reports, and multi-issue
reports where the precedence rule decides the category.
"""

import csv
import random
from pathlib import Path

SEED = 2049
OUT = Path(__file__).parent

# ---------------------------------------------------------------- slot values
SITES = ["Rotterdam DC", "Leeds hub", "Gate 4 at Felixstowe", "Dock 7", "the Milton Keynes cross-dock",
         "Warehouse B", "Antwerp terminal", "the Manchester depot", "Bay 12", "the Dublin yard",
         "Hamburg port", "the M6 services near Stafford", "Coventry DC", "the Tilbury berth"]
CARGO = ["pallets of electronics", "furniture", "auto parts", "retail clothing", "bottled drinks",
         "paper rolls", "garden tools", "white goods", "books", "toys", "printer cartridges"]
PERISHABLE = ["vaccines", "insulin", "fresh fish", "blood samples", "chilled dairy", "frozen chicken",
              "cut flowers", "clinical trial kits"]
HAZMAT = ["lithium batteries", "paint thinner", "chlorine tablets", "diesel", "aerosol cans", "battery acid"]
CUSTOMERS = ["Northwind", "Contoso Retail", "Fabrikam", "Tailspin Medical", "Globex", "Initech",
             "Acme Foods", "Umbrella Pharma", "Wayne Hardware", "Stark Home"]
VEHICLES = ["truck 14", "trailer TR-221", "van 3", "forklift F2", "the reach truck", "unit 88",
            "the tractor unit", "artic 40"]
EQUIP = ["conveyor line 2", "the handheld scanners", "dock leveller 5", "the WMS label printer",
         "the sorter", "the reefer unit on trailer 9"]
REPORTERS = ["", "Driver report: ", "Email from customer: ", "Ops note - ", "Shift lead: ",
             "Call log: ", "From the yard: ", "Ticket via portal: ", "WhatsApp from driver: "]


def slot(rng):
    return {
        "site": rng.choice(SITES), "cargo": rng.choice(CARGO), "perish": rng.choice(PERISHABLE),
        "haz": rng.choice(HAZMAT), "cust": rng.choice(CUSTOMERS), "veh": rng.choice(VEHICLES),
        "equip": rng.choice(EQUIP), "ship": f"SHP-{rng.randint(10000, 99999)}",
        "n": rng.randint(2, 40), "hrs": rng.randint(2, 9), "mins": rng.choice([20, 30, 45, 50]),
        "temp": rng.choice([9, 11, 12, 14, 15]), "day": rng.choice(["Monday", "Tuesday", "yesterday", "last week", "Friday"]),
    }


# ---------------------------------------------------------------- templates
# (category, urgency, difficulty, template). difficulty: easy | hard
T = []


def add(cat, urg, templates, difficulty="easy"):
    for t in templates:
        T.append((cat, urg, difficulty, t))


add("delivery_delay", "normal", [
    "Shipment {ship} for {cust} is running about {hrs} hours late due to traffic. Customer informed, new ETA tonight.",
    "Missed our unloading slot at {site}, rebooked for tomorrow morning. {cargo} still on the trailer, all fine.",
    "Delivery to {cust} pushed to {day} because the consignee closed early. No damage, just need a new slot.",
    "Queue at {site} is long, driver waiting {hrs}h. Load is {cargo}, not time critical.",
    "Hi, my order {ship} was meant to arrive {day} and hasn't come. Can someone give me an update please?",
    "Ferry cancelled, trailer with {cargo} will cross on the next sailing. ETA slips by {hrs} hours.",
    "{cust} asked to postpone delivery of {ship} to next week, need to rebook the carrier.",
])
add("delivery_delay", "urgent", [
    "{perish} for {cust} stuck in traffic, delivery window closes in {mins} minutes and the hospital has no stock left.",
    "Trailer jack-knifed ahead, our truck carrying {perish} is stuck and the cold pack only lasts another hour. Need a rescue vehicle now.",
    "Same-day {perish} run for {cust} is 3 hours behind and the surgery is scheduled for this afternoon. Need a courier option now.",
], "easy")
add("delivery_delay", "normal", [
    "THIS IS UNACCEPTABLE!!! Third time {ship} is late. I want a call TODAY from a manager!!!",
    "URGENT URGENT - customer {cust} chasing ETA on {ship} again, they are very annoyed. Delivery due {day}, just slow.",
    "Escalating!! Our {cargo} delivery was 2 hours late and the driver was rude. Please sort this out.",
], "hard")  # angry tone, but not operationally urgent
add("delivery_delay", "urgent", [
    "Small thing, but the {perish} for {cust} hasn't left {site} yet and it must be with the clinic by 14:00. It's 13:10.",
    "fyi load {ship} is at {site} still. It's the {perish} order that the patient needs this afternoon.",
], "hard")  # calm tone, but urgent

add("damaged_goods", "normal", [
    "Received {ship} with {n} cartons crushed, {cargo} inside look ok but packaging is ruined. Photos attached.",
    "Pallet of {cargo} dropped during unloading at {site}, about {n} units broken. No one hurt.",
    "Customer {cust} reports water damage on {n} boxes of {cargo}. Delivered {day}, raising a claim.",
    "Shrink wrap torn and two layers of {cargo} have shifted, some dents. Accepted with remarks.",
    "Item arrived scratched, the box was clearly mishandled. Want a refund for {ship}.",
    "Forklift tines punctured {n} cartons of {cargo} in the racking. Stock quarantined for inspection.",
    "Rain got into trailer through a roof tear, {cargo} wet. Insurance claim needed.",
])
add("damaged_goods", "urgent", [
    "Pallet collapsed at {site} and {cargo} is spread across the main yard road, trucks can't get in or out.",
    "Crate of glass fell off the trailer onto lane 2 of the access road, road blocked, police asking for us to clear it now.",
], "easy")
add("damaged_goods", "normal", [
    "DISGUSTING SERVICE. My {cargo} arrived smashed to pieces!!! I need this fixed ASAP or I go to the press.",
    "Box of {haz} arrived dented but sealed, nothing leaking, checked by the team. Logging for the claim.",
], "hard")

add("lost_or_stolen", "normal", [
    "{ship} shows delivered but {cust} says nothing arrived. Need proof of delivery checked.",
    "{n} cartons missing from the pallet count at {site} compared to the manifest. Possibly short-shipped.",
    "Parcel not scanned since leaving {site} {day}. Trace requested.",
    "Wrong pallet delivered to {cust}, their {cargo} must have gone to another address.",
    "Seal number on trailer doesn't match paperwork but load count is complete. Logging it.",
    "Customer says package left at the wrong house, neighbours don't have it.",
])
add("lost_or_stolen", "urgent", [
    "Driver reports someone is cutting the curtain of trailer at {site} right now, he is watching from the cab. Load is {cargo}.",
    "Truck {veh} with full load of {cargo} has been taken from the lay-by at {site}, tracker shows it moving now.",
    "Break-in at Warehouse B happening, alarm going off and two people seen loading {cargo} into a van.",
    "GPS tracker on {ship} ({perish}) just went dark and the truck is off route by 30km. Driver not answering.",
])
add("lost_or_stolen", "normal", [
    "Pallet missing!!! Where is my stuff?!?! {ship} was due {day}. Answer me!!",
    "Looks like {n} boxes of {cargo} were stolen from the yard over the weekend, found the fence cut this morning. Police report done.",
], "hard")  # past theft, already reported -> normal

add("safety_incident", "urgent", [
    "Worker injured at {site}, forklift ran over his foot. Ambulance called.",
    "Leak of {haz} from a damaged drum on Dock 7, strong smell, staff evacuated.",
    "Fire in the battery charging area at {site}, fire brigade on the way.",
    "Driver collapsed in the cab at {site}, first aider with him, not responding well.",
    "Racking at {site} is leaning badly after a truck hit it, area not yet cordoned and people still picking.",
    "Spill of {haz} on the motorway hard shoulder from our trailer, highways agency on scene.",
    "Colleague fell from the loading dock edge, bleeding from the head.",
])
add("safety_incident", "normal", [
    "Near miss {day}: reversing truck came close to a pedestrian at {site}. No injury. Suggest a banksman.",
    "Toolbox talk follow-up: hi-vis compliance was poor on night shift last week.",
    "Minor cut to a picker's finger from a box cutter {day}, plaster applied, back at work. Logging per policy.",
    "Fire extinguisher at Bay 12 expired last month, needs replacing.",
    "Monthly safety audit at {site} found two blocked fire exits, now cleared.",
], "easy")
add("safety_incident", "urgent", [
    "Hey team, sorry to bother, small one: a pallet of {haz} on Dock 7 is leaking a bit onto the floor and a couple of folks feel dizzy.",
    "Damaged boxes of {haz} received, one is smoking slightly. Have put it outside.",
    "Truck {veh} has failed brakes and rolled into the gatehouse, the guard is trapped inside.",
], "hard")  # calm wording, or multi-issue where safety takes precedence

add("vehicle_equipment_failure", "normal", [
    "{veh} has a flat tyre at {site}, recovery booked, load can wait until tomorrow.",
    "{equip} down since this morning, picking manually. Engineer coming at 3pm.",
    "Warning light on the dashboard of {veh}, booked into the workshop {day}.",
    "Tail lift on van 3 is slow to lower, still usable, please schedule a service.",
    "{equip} keeps jamming every hour or so, maintenance ticket raised.",
    "AdBlue fault on {veh}, limp mode, driver heading back to depot at low speed.",
])
add("vehicle_equipment_failure", "urgent", [
    "{veh} broke down in the middle lane of the M6, hazard lights on, traffic building behind it.",
    "Main sorter at {site} has stopped completely during peak, 4000 parcels for same-day dispatch stuck.",
    "Reefer unit on trailer 9 died, trailer full of {perish}, temp already rising. Need a replacement trailer.",
])
add("vehicle_equipment_failure", "normal", [
    "BROKEN AGAIN!!! {equip} is a joke. Fix it properly this time!!!",
    "{veh} wont start, its the spare one, nobody needs it this week.",
], "hard")

add("documentation_customs", "normal", [
    "Customs hold on {ship} at {site}, missing commercial invoice. Broker asking for a copy.",
    "Packing list quantity doesn't match the invoice for {cust}, please confirm the correct figure.",
    "EORI number missing on the declaration for {ship}.",
    "Delivery note for {cust} has the wrong address, goods delivered correctly though.",
    "Certificate of origin not attached, preferential duty can't be claimed.",
    "Labels printed with the wrong barcode on {n} cartons, relabelling at {site}.",
    "Need ADR paperwork updated for {haz} before the {day} shipment.",
])
add("documentation_customs", "urgent", [
    "Border force holding trailer of {perish} at {site} for missing health certificate. Product spoils in 2 hours.",
    "Driver at {site} without ADR documents for {haz}, police have stopped the truck on the roadside and will impound it.",
])
add("documentation_customs", "normal", [
    "ASAP!!! Invoice for {ship} is wrong, we were overcharged. Fix it!!!",
], "hard")

# Changed-requirement scenario: new category, mostly unseen phrasing.
add("temperature_excursion", "urgent", [
    "Datalogger on {ship} shows {perish} at {temp}°C for the last 40 minutes, limit is 8°C. Still in transit.",
    "Cold room door at {site} left open overnight, {perish} reading {temp} degrees. Product at risk.",
    "Reefer alarm on the trailer: setpoint 2-8C, current {temp}C, load is {perish} for {cust}.",
    "Temp excursion! {perish} pallet sat on the warm dock for {hrs}h before being put in the chiller.",
])
add("temperature_excursion", "normal", [
    "Post-delivery review: logger for {ship} recorded a 6 minute spike to {temp}°C {day}. Within stability budget, QA informed.",
    "Monthly cold-chain report: two minor excursions on the {perish} lane, both released by QA.",
    "Calibration of the temperature probes at {site} is overdue.",
    "{cust} asking for the temperature log PDF for {ship}, delivered {day}.",
])

# Extra v1 phrasings reserved for v2 (tests generalisation to new wording).
V2_ONLY_V1 = [
    ("delivery_delay", "normal", "hard", "Consignment {ship} is held up behind a strike at {site}. Rescheduling with {cust}."),
    ("damaged_goods", "normal", "hard", "Box of {cargo} damaged during transit, customer wants replacement."),
    ("lost_or_stolen", "urgent", "hard", "Someone just drove off with our trailer from {site}, the yard CCTV shows it now."),
    ("safety_incident", "urgent", "hard", "Strong smell of gas near Dock 7, two staff feeling sick, we have stopped work."),
    ("vehicle_equipment_failure", "normal", "hard", "Pallet jack at {site} handle snapped, using the other one."),
    ("documentation_customs", "normal", "hard", "Import licence for {cust} expires next month, please renew."),
    ("temperature_excursion", "urgent", "hard", "Chiller at {site} reading {temp}C and climbing, {perish} inside. Engineer not answering."),
]

TYPOS = {"the": "teh", "received": "recieved", "delivery": "delivey", "because": "becuase", "warehouse": "warehose"}


def noisify(text, rng):
    if rng.random() < 0.15:
        for a, b in TYPOS.items():
            if a in text and rng.random() < 0.5:
                text = text.replace(a, b, 1)
    if rng.random() < 0.1:
        text = text.lower()
    return rng.choice(REPORTERS) + text


def render(rng, template):
    return noisify(template.format(**slot(rng)), rng)


def main():
    rng = random.Random(SEED)
    v1_templates = [t for t in T if t[0] != "temperature_excursion"]

    rows, seen = [], set()
    # Balance: every template at least once, then sample to 360 weighted toward easy.
    pool = list(v1_templates)
    while len(pool) < 360:
        pool.append(rng.choice(v1_templates))
    rng.shuffle(pool)
    for cat, urg, diff, tpl in pool:
        for _ in range(10):
            text = render(rng, tpl)
            if text not in seen:
                break
        seen.add(text)
        rows.append({"text": text, "category": cat, "urgency": urg, "difficulty": diff,
                     "template_id": T.index((cat, urg, diff, tpl))})

    # Split by TEMPLATE, stratified by (category, urgency): a test report's wording never
    # appears in train. A random row split leaks templates and inflates TF-IDF to ~99%.
    by_key = {}
    for i, (cat, urg, _, _) in enumerate(T):
        if cat != "temperature_excursion":
            by_key.setdefault((cat, urg), []).append(i)
    template_split = {}
    for key in sorted(by_key):
        ids = by_key[key]
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * 0.25)) if len(ids) >= 2 else 0
        n_val = max(1, round(len(ids) * 0.15)) if len(ids) >= 3 else 0
        for j, tid in enumerate(ids):
            template_split[tid] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    for r in rows:
        r["split"] = template_split[r["template_id"]]
    # Secondary random split (same 60/15/25 proportions), stratified by (category, urgency):
    # measures performance on wording similar to history.
    shuffled = rows[:]
    rng.shuffle(shuffled)
    groups = {}
    for r in shuffled:
        groups.setdefault((r["category"], r["urgency"]), []).append(r)
    for key in sorted(groups):
        g = groups[key]
        n_test, n_val = round(len(g) * 0.25), round(len(g) * 0.15)
        for j, r in enumerate(g):
            r["split_random"] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    rows.sort(key=lambda r: (r["split"], r["category"]))
    for i, r in enumerate(rows):
        r["id"] = f"INC-{i + 1:04d}"

    fields = ["id", "split", "split_random", "text", "category", "urgency", "difficulty", "template_id"]
    with open(OUT / "incidents.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # v2: new category + fresh phrasings of old categories.
    temp_t = [t for t in T if t[0] == "temperature_excursion"]
    v2 = []
    for i in range(40):
        cat, urg, diff, tpl = temp_t[i % len(temp_t)] if i < 36 else V2_ONLY_V1[-1]
        v2.append({"text": render(rng, tpl), "category": cat, "urgency": urg, "difficulty": diff, "template_id": -1})
    for i in range(40):
        # Reefer-failure template is excluded: under the v2 taxonomy it is a temperature_excursion.
        cat, urg, diff, tpl = V2_ONLY_V1[i % 6] if i < 12 else rng.choice(
            [t for t in v1_templates if "Reefer" not in t[3]])
        v2.append({"text": render(rng, tpl), "category": cat, "urgency": urg, "difficulty": diff, "template_id": -1})
    rng.shuffle(v2)
    for i, r in enumerate(v2):
        r["id"] = f"V2-{i + 1:04d}"
        r["split"] = r["split_random"] = "v2"
    with open(OUT / "incidents_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(v2)

    print(f"wrote {len(rows)} v1 reports and {len(v2)} v2 reports from {len(T)} templates")


if __name__ == "__main__":
    main()
