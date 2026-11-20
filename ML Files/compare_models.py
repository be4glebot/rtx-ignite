"""Compare the random forest, gradient boosting, and their average on the same splits as train.py."""
import gzip, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from simulate import FEATURES, HELD
df = pd.read_csv("features.csv")
rng = np.random.default_rng(7); ids = np.array(df.case_id.unique()); rng.shuffle(ids); tc = set(ids[:int(len(ids) * 0.2)])
test = df[df.case_id.isin(tc) & ~df.scenario_id.isin(HELD)]; ho = df[df.scenario_id.isin(HELD)]
rf, gb = pickle.load(gzip.open("model.pkl.gz", "rb")).models     # train.py with MODEL=avg saves both
cls = list(rf.classes_); assert cls == list(gb.classes_)
def probs(name, d):
    if name == "rf": return rf.predict_proba(d[FEATURES])
    if name == "gb": return gb.predict_proba(d[FEATURES])
    return 0.5 * rf.predict_proba(d[FEATURES]) + 0.5 * gb.predict_proba(d[FEATURES])
for name in ("rf", "gb", "avg"):
    Pt, Ph = probs(name, test), probs(name, ho)
    pt = np.array(cls)[Pt.argmax(1)]; ph = np.array(cls)[Ph.argmax(1)]
    t = test.assign(p=pt, ok=pt == test.label.values); h = ho.assign(p=ph, ok=ph == ho.label.values)
    thr = t[t.label == "threat"]; fa = t[t.label.isin(["authorized", "non_drone"])]
    ut, uh = 1 - Pt.max(1), 1 - Ph.max(1)
    print("%-4s test %.3f | per-scenario %.3f | threat recall/scen %.3f | at 5 s %.3f | false threat %.4f | held-out rows %.3f, per-scenario %.3f | unfamiliar AUC %.3f, wrong %.2f vs right %.2f" % (
        name, t.ok.mean(), t.groupby("scenario_id").ok.mean().mean(), thr.groupby("scenario_id").ok.mean().mean(),
        thr[thr.checkpoint_s == 5].ok.mean(), (fa.p == "threat").mean(), h.ok.mean(), h.groupby("scenario_id").ok.mean().mean(),
        roc_auc_score([0] * len(ut) + [1] * len(uh), list(ut) + list(uh)), ut[~t.ok.values].mean(), ut[t.ok.values].mean()))
    print("      held-out by scenario:", h.groupby("scenario_id").ok.mean().round(2).to_dict())
