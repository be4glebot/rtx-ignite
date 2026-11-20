"""Measured timing: (1) compute time of fusion features + model per track update on this laptop-class CPU,
(2) seconds from first sensor detection to first operator notification / alert / alarm in the simulation (threats from outside)."""
import time, gzip, pickle, numpy as np, pandas as pd, simulate as S
from build_demo import run_case
clf = pickle.load(gzip.open("model.pkl.gz", "rb"))
for m in clf.models:
    if hasattr(m, "n_jobs"): m.n_jobs = 1
cases = pd.read_csv("cuas_test_cases.csv"); st = pd.read_csv("case_stats.csv")
thr = st[(st.label == "threat") & (st.starts_inside == 0) & (st.detected > 0)].case_id
pool = cases[cases.case_id.isin(thr)].sample(150, random_state=7)
# (1) compute time
row = pool.iloc[0].to_dict(); rs = np.random.default_rng(int(row["seed"]))
objs, _ = S.build_case(row, rs); objs = S.simulate_sensors(row, objs, rs)
i = next(k for k, o in enumerate(objs) if o["first"] >= 0); t = objs[i]["first"] + 10
N = 200; t0 = time.perf_counter()
for _ in range(N): f = S.features_at(row, objs, i, t, rs)
tf = (time.perf_counter() - t0) / N
X1 = pd.DataFrame([f])[S.FEATURES]; X50 = pd.concat([X1] * 50, ignore_index=True)
clf.predict_proba(X1)
t0 = time.perf_counter(); [clf.predict_proba(X1) for _ in range(30)]; tm1 = (time.perf_counter() - t0) / 30
t0 = time.perf_counter(); [clf.predict_proba(X50) for _ in range(10)]; tm50 = (time.perf_counter() - t0) / 10
print(f"features per track update: {tf*1000:.1f} ms | model, 1 track: {tm1*1000:.0f} ms | model, 50 tracks in one batch: {tm50*1000:.0f} ms")
# (2) decision timing in the simulation
rows = []
for _, r in pool.iterrows():
    tracks, _ = run_case(r.to_dict(), clf)
    for tr in tracks:
        if tr["bg"]: continue
        lv = np.array(tr["lvl"]); d0 = tr["first"]
        f = lambda L: (int(np.argmax(lv >= L)) - d0) if (lv >= L).any() else None
        rows.append(dict(watch=f(1), alert=f(2), alarm=f(3)))
d = pd.DataFrame(rows)
for c in ("watch", "alert", "alarm"):
    v = d[c].dropna()
    print(f"first {c:5s}: reached in {len(v)/len(d):.0%} of {len(d)} threat tracks | seconds after first detection: median {v.median():.0f}, 10th pct {v.quantile(.1):.0f}, 90th pct {v.quantile(.9):.0f}")
