"""Second-by-second check of the hold-steady rule on cases the model never trained on."""
import gzip, pickle, numpy as np, pandas as pd
from build_demo import run_case, unseen_ids
from simulate import HELD
clf = pickle.load(gzip.open("model.pkl.gz", "rb")); cases = pd.read_csv("cuas_test_cases.csv"); U = unseen_ids()
pool = cases[cases.case_id.isin(U) | cases.scenario_id.isin(HELD)].sample(300, random_state=3)
# add every unseen case of the scenarios with a behaviour change, so the alert-time numbers have a real sample
pool = pd.concat([pool, cases[cases.case_id.isin(U) & cases.scenario_id.isin([26, 47, 77, 80])]]).drop_duplicates("case_id")
tot = dict(total=0, right=0, right_win=0, raw_right=0, flips_raw=0, flips_steady=0); n = 0; EV = []; CW = []
for _, r in pool.iterrows():
    _, st = run_case(r.to_dict(), clf)
    if st["total"]:
        n += 1
        for k in tot: tot[k] += st[k]
        EV += st["events"]; CW += st["cam_waits"]
print("cases %d, track-seconds %d" % (n, tot["total"]))
print("right call, per second: raw model %.1f%%, with hold-steady rule %.1f%%" % (100 * tot["raw_right"] / tot["total"], 100 * tot["right"] / tot["total"]))
print("label changes per case: raw %.1f, with hold-steady rule %.1f" % (tot["flips_raw"] / n, tot["flips_steady"] / n))
print("right call with the transition window (early warning before a behaviour change counts): %.1f%%" % (100 * tot["right_win"] / tot["total"]))
ev = pd.DataFrame(EV)
if len(ev):
    print("\nBehaviour changes (hijack, drift, breakout): %d tracks" % len(ev))
    for sid, g in ev.groupby("sid"):
        a = g.alert.dropna(); th = g.threat.dropna()
        print("  scenario %d: operator alerted in %d of %d (median %+.0f s from the change, %d before it)%s" % (
            sid, len(a), len(g), a.median() if len(a) else float("nan"), (a < 0).sum(),
            "; reached threat in %d (median %+.0f s)" % (len(th), th.median()) if g.to.iloc[0] == "threat" and len(th) else ""))
cw = pd.DataFrame(CW, columns=["multi", "wait"])
for m, g in cw.groupby("multi"):
    w = g.wait.dropna()
    print("camera wait once in range, %s: median %.0f s, 90th pct %.0f s, never looked at %d of %d" % ("several objects" if m else "single drone", w.median(), w.quantile(0.9), g.wait.isna().sum(), len(g)))
