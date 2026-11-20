"""Turn gameday_tracks.csv into per-encounter alarm probabilities and expected alarms per game."""
import numpy as np, pandas as pd
from gameday_fp import RATES
df = pd.read_csv("gameday_tracks.csv")
# one encounter = one case (a flock of 12 birds alarming together is one incident for the operator)
import sys
if len(sys.argv) > 1 and sys.argv[1] == "nogate": df["ever_threat"] = df["ever_threat_nogate"]; print("WITHOUT the alert rule")
if "ever_notified" in df: df["ever_alert"] = df["ever_notified"]      # operator actually notified (alert or alarm level), not just "watching"
enc = df.groupby(["case_id", "cat"]).agg(ever_threat=("ever_threat", "max"), ever_alert=("ever_alert", "max"), label=("label", "first")).reset_index()
benign = lambda d: d[d.label.isin(["authorized", "non_drone"])]
print("tracks %d, encounters %d" % (len(df), len(enc)))
tab = enc.groupby("cat").agg(n=("ever_threat", "size"), p_threat=("ever_threat", "mean"), p_alert=("ever_alert", "mean"))
trk = df.groupby("cat").agg(tracks=("ever_threat", "size"), trk_threat=("ever_threat", "mean"), threat_secs=("threat_s", "mean"))
tab = tab.join(trk)
tot = {k: [0.0, 0.0, 0.0] for k in ("threat", "alert", "over")}
rows = []
for cat, r in tab.iterrows():
    lo, ty, hi = RATES[cat]
    careless = cat.startswith("Hobbyist")
    for i, rate in enumerate((lo, ty, hi)):
        if careless: tot["over"][i] += rate * r.p_threat
        else: tot["threat"][i] += rate * r.p_threat; tot["alert"][i] += rate * r.p_alert
    rows.append((cat, int(r.n), r.p_threat, (r.p_alert if not careless else np.nan), ty, ty * r.p_threat))
out = pd.DataFrame(rows, columns=["traffic", "encounters simulated", "P(threat alarm)", "P(operator notified)", "typical per game", "threat alarms per game (typical)"])
pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))
print("\nExpected per game (low / typical / high traffic):")
print("  false THREAT alarms on authorized or non-drone traffic: %.1f / %.1f / %.1f" % tuple(tot["threat"]))
print("  operator notified about authorized or non-drone traffic: %.1f / %.1f / %.1f" % tuple(tot["alert"]))
print("  careless hobbyists escalated to THREAT (over-escalation): %.1f / %.1f / %.1f" % tuple(tot["over"]))
