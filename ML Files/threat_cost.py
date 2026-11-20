"""What the alert rule costs on real threats: replay threat cases the model never trained on and compare
alarm recall and warning time (alarm -> drone reaches the bowl) with and without the rule."""
import gzip, pickle, sys, numpy as np, pandas as pd
from build_demo import run_case, unseen_ids
from simulate import HELD
clf = pickle.load(gzip.open("model.pkl.gz", "rb")); clf.models[0].n_jobs = 1; cases = pd.read_csv("cuas_test_cases.csv"); U = unseen_ids()
pool = cases[(cases.case_id.isin(U) | cases.scenario_id.isin(HELD)) & (cases.expected_label == "threat")]
pool = pool.sample(min(int(sys.argv[1]) if len(sys.argv) > 1 else 400, len(pool)), random_state=1)
recs = []
for _, r in pool.iterrows():
    row = r.to_dict(); tracks, st = run_case(row, clf)
    for tr in tracks:
        if tr["bg"] or tr["truth"][-1] != 4 and 4 not in tr["truth"]: continue
        d = [None if x is None else (x * x + y * y) ** 0.5 for x, y in zip(tr["tx"], tr["ty"])]
        reach = next((k for k, v in enumerate(d) if v is not None and v < 120 and k >= tr["first"]), None)
        start_in = d[tr["first"]] is not None and d[tr["first"]] < 150
        out = dict(sid=int(row["scenario_id"]), reach=reach, start_inside=start_in)
        for key, name in (("predg", "gate"), ("pred", "nogate")):
            a = next((k for k, p in enumerate(tr[key]) if p == 4), None)
            al = next((k for k, p in enumerate(tr[key]) if p in (3, 4)), None)      # any operator alert (borderline or threat)
            out[name + "_alarm"] = a; out[name + "_alert"] = al
        out["notified"] = next((k for k, v in enumerate(tr["lvl"]) if v >= 2), None)
        recs.append(out)
df = pd.DataFrame(recs); df.to_csv("threat_cost_tracks.csv", index=False)
print("threat tracks: %d (%d cases)" % (len(df), len(pool)))
for name in ("nogate", "gate"):
    a = df[name + "_alarm"]; al = df[name + "_alert"]
    o = df[~df.start_inside & df.reach.notna()]
    w = (o.reach - o[name + "_alarm"]).where(o[name + "_alarm"].notna() & (o[name + "_alarm"] <= o.reach))
    wa = (o.reach - o[name + "_alert"]).where(o[name + "_alert"].notna() & (o[name + "_alert"] <= o.reach))
    late = (o[name + "_alarm"].isna() | (o[name + "_alarm"] > o.reach)).mean()
    print("%-7s threat ALARM ever: %.1f%% | operator alerted ever: %.1f%% | outside threats that reach the bowl: alarm before arrival %.1f%%, warning from alarm median %.0f s (10th pct %.0f s), from first alert median %.0f s" % (
        "without" if name == "nogate" else "with", 100 * a.notna().mean(), 100 * al.notna().mean(), 100 * (1 - late), w.median(), w.quantile(0.1), wa.median()))
o = df[~df.start_inside & df.reach.notna()]
nb = (o.notified.notna() & (o.notified <= o.reach)).mean(); wn = (o.reach - o.notified).where(o.notified.notna() & (o.notified <= o.reach))
print("with the watch tier: operator notified before arrival %.1f%%, warning from notification median %.0f s (10th pct %.0f s); notified at some point %.1f%% of all threat tracks" % (100 * nb, wn.median(), wn.quantile(0.1), 100 * df.notified.notna().mean()))
