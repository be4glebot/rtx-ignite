"""Fresh benign cases for the false-alarm study: same scenario definitions as gen_cases.py,
but a different random seed, so none of these flights were seen in training."""
import csv, random
src = open("gen_cases.py").read()
src = src[:src.index("def add(r):")].replace("random.Random(20261116)", "random.Random(5150)")
ns = {}; exec(src, ns)
N = {**{s: 25 for s in (17, 18, 19, 20, 21, 22, 23, 25, 36)}, **{s: 15 for s in (1, 2, 3, 4, 6, 68, 69, 70, 71, 72, 73)},
     5: 20, 7: 100, 8: 20, 9: 60, 10: 60, 11: 60, 12: 25, 13: 25, 14: 25, 15: 60, 16: 60}
rows = []
for sc in ns["S"]:
    for _ in range(N.get(sc[0], 0)): rows.append(ns["make"](sc))
for i, r in enumerate(rows): r["case_id"] = "GD-%04d" % (i + 1)
with open("gameday_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ns["FIELDS"]); w.writeheader(); w.writerows(rows)
print(len(rows), "fresh benign cases")
