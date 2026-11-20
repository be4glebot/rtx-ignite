"""Cases for the swarm study: hostile groups, benign groups, and three new scenarios that only
coordination can explain. New random seed, so none of these flights were seen in training.
The new scenarios (81-83) are defined here, not in gen_cases.py, so the training set is unchanged."""
import csv
src = open("gen_cases.py").read()
src = src[:src.index("def add(r):")].replace("random.Random(20261116)", "random.Random(8128)")
ns = {}; exec(src, ns)
NEW = [
    (81, "E", "Formation of drones holding 600 to 1,200 m out, watching the crowd", "borderline", "alert_operator",
     dict(n=(3, 8), range=(2000, 4500), alt=(40, 120), speed=(6, 15), path="formation_hold", closest=(600, 1200), rf=[True, False], rid="none", listed=False, target="none", swarm=True)),
    (82, "E", "Swarm holds in formation, then attacks together", "threat", "priority_alert",
     dict(n=(3, 8), range=(1200, 2600), alt=(40, 120), speed=(8, 18), path="formation_hold_attack", closest=(600, 1200), rf=[True, False], rid="none", listed=False, target="bowl", swarm=True)),
    (83, "C", "Several unrelated hobbyists flying near the same tailgate lot", "unauthorized_careless", "dispatch_police_to_pilot",
     dict(n=(3, 5), range=(300, 700), alt=(15, 60), speed=(2, 8), path="tailgate_cluster", closest=(150, 700), rf=True, rid=["valid", "none"], listed=False, target="none")),
]
S = {s[0]: s for s in ns["S"] + NEW}
HOSTILE = {51: 40, 52: 40, 53: 40, 75: 30, 81: 60, 82: 60, 76: 40, 54: 30, 55: 30, 56: 30, 49: 30}
BENIGN = {5: 20, 68: 20, 73: 40, 4: 40, 9: 60, 11: 40, 83: 60}
rows = []
for sid, k in list(HOSTILE.items()) + list(BENIGN.items()):
    c = 0
    while c < k:
        r = ns["make"](S[sid])
        if sid in (54, 55, 56) and r["n_objects"] < 2: continue          # multi-drone versions only
        if sid == 49 and r["n_objects"] < 3: continue                     # decoy plus at least two attackers
        if sid == 4: r["n_objects"] = 2                                    # police pair
        rows.append(r); c += 1
over = ns["S"][16]                                                         # scenario 17: hobbyist over a lot
for _ in range(40):                                                        # ten or more hobbyists at once (independent)
    r = ns["make"](over); r["concurrent_event"] = "%d_other_hobbyists" % ns["R"].randint(10, 25); r["scenario_id"] = 17; rows.append(r)
for i, r in enumerate(rows): r["case_id"] = "SW-%04d" % (i + 1)
with open("swarm_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ns["FIELDS"]); w.writeheader(); w.writerows(rows)
print(len(rows), "swarm-study cases")
