import numpy as np, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent          # this folder (site_survey/)
D = pickle.load(open(HERE / "vis.pkl", "rb")); S = D["sites"]; VIS = D["VIS"]; W = D["WGT"]; TR = D["TR"]
near = TR <= 600; far = TR > 600
def cov(m, sel=None): sel = np.ones(len(W), bool) if sel is None else sel; return float((W*m)[sel].sum()/W[sel].sum())
def pair_stats(ids):
    out = {}
    for a in (15, 30, 60):
        v = VIS[a][list(ids)]; n = v.sum(0)
        out[a] = (cov(n >= 1, near), cov(n >= 1, far), cov(n >= 1), cov(n >= 2), min(cov(np.delete(v, i, 0).sum(0) >= 1) for i in range(len(ids))))
    return out
N = len(S); nt = [i for i, s in enumerate(S) if s["kind"] != "trailer"]
# score all pairs: mean of 15 m and 30 m coverage, plus 0.5 x one-down at 30 m
V15, V30 = VIS[15].astype(np.float32), VIS[30].astype(np.float32)
c15 = V15 @ W; c30 = V30 @ W
best = []
for i in range(N):
    u15 = c15[i] + c15 - (V15[i] * V15) @ W; u30 = c30[i] + c30 - (V30[i] * V30) @ W
    od30 = np.minimum(c30[i], c30)
    sc = 0.5*(u15 + u30) + 0.5*od30
    for j in np.argsort(-sc)[:5]:
        if j != i: best.append((float(sc[j]), min(i, j), max(i, j)))
best = sorted(set(best), reverse=True)
def show(title, ids):
    st = pair_stats(ids)
    print(f"\n{title}: " + " + ".join(f"{S[i]['name']} ({S[i]['z']:.0f} m)" for i in ids))
    for a in (15, 30, 60):
        nr, fr, al, two, od = st[a]
        print(f"   drone {a:2d} m up: seen {100*nr:4.0f}% within 600 m, {100*fr:4.0f}% at 600-1300 m | by 2 radars {100*two:4.0f}% | one radar down {100*od:4.0f}%")
print("TOP PAIRS, any site type:")
seen = set(); k = 0
for sc, i, j in best:
    if (i, j) in seen: continue
    seen.add((i, j)); show(f"#{k+1}", (i, j)); k += 1
    if k == 5: break
print("\nTOP PAIRS using only rooftops/garages/stadium (no trailers):")
roof_pairs = sorted(((0.5*(cov(VIS[15][[i, j]].any(0)) + cov(VIS[30][[i, j]].any(0))) + 0.5*min(cov(VIS[30][i]), cov(VIS[30][j])), i, j) for a_, i in enumerate(nt) for j in nt[a_+1:]), reverse=True)
for r, (sc, i, j) in enumerate(roof_pairs[:6]): show(f"R{r+1}", (i, j))
idx = {s["name"]: i for i, s in enumerate(S)}
show("BASELINE current design (2 corner roofs, diagonal)", (idx["Stadium N-W corner roof"], idx["Stadium S-E corner roof"]))
show("BASELINE both canopy tops", (idx["Stadium canopy top (west)"], idx["Stadium canopy top (east)"]))
print("\nSingle sites ranked by coverage at 30 m (rooftops):")
for i in sorted(nt, key=lambda i: -cov(VIS[30][i])): print(f"   {S[i]['name']:45s} {S[i]['z']:4.0f} m  {100*cov(VIS[30][i]):4.0f}% (15 m: {100*cov(VIS[15][i]):3.0f}%, 60 m: {100*cov(VIS[60][i]):3.0f}%)  at {np.hypot(S[i]['x'],S[i]['y']):.0f} m brg {np.degrees(np.arctan2(S[i]['x'],S[i]['y']))%360:.0f}")
