"""Workshop advice: 'Start with clear rules. Then upgrade one block with ML and compare.'
Compares, per snapshot, a rules-only alarm with the model alarm on the SAME test and held-out snapshots.
  rules-only alarm: drone confirmed AND urgent AND not consistent with an approved flight
  model alarm:      model says THREAT AND drone confirmed AND urgent (our actual alarm rule)
Snapshot level, no persistence or latching, so rates differ from the track-level game-day numbers."""
import pandas as pd, numpy as np, pickle, gzip, alert_rules as ar
from simulate import FEATURES, HELD
df = pd.read_csv("features.csv")
rng = np.random.default_rng(7)  # must match train.py
cases = np.array(df.case_id.unique()); rng.shuffle(cases); test_cases = set(cases[:int(len(cases) * 0.2)])
held = df.scenario_id.isin(HELD)
sets = {"test (seen types)": df[~held & df.case_id.isin(test_cases)], "held-out types": df[held]}
clf = pickle.load(gzip.open("model.pkl.gz"))
for name, d in sets.items():
    recs = d.to_dict("records")
    conf = np.array([ar.confirmed(r) for r in recs]); urg = np.array([ar.urgent(r) for r in recs])
    unapproved = (d.on_manifest.values == 0) | (d.outside_plan_s.values > 0) | (d.manifest_time_ok.values == 0)
    rule = conf & urg & unapproved
    model = conf & urg & (clf.predict(d[FEATURES]) == "threat")
    print(f"\n{name}: {len(d)} snapshots, {d.case_id.nunique()} cases")
    print(f"{'true label':24s}{'n':>7s}{'rules alarm %':>15s}{'model alarm %':>15s}")
    for lab in ["threat", "borderline", "unauthorized_careless", "authorized", "non_drone"]:
        m = d.label.values == lab
        if m.sum(): print(f"{lab:24s}{m.sum():7d}{rule[m].mean()*100:15.1f}{model[m].mean()*100:15.1f}")
