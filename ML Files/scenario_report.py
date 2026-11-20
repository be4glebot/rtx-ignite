import gzip, pickle, numpy as np, pandas as pd
from simulate import FEATURES, HELD
df=pd.read_csv("features.csv"); clf=pickle.load(gzip.open("model.pkl.gz","rb"))
rng=np.random.default_rng(7); cases=np.array(df.case_id.unique()); rng.shuffle(cases); tc=set(cases[:int(len(cases)*0.2)])
t=df[~df.scenario_id.isin(HELD)&df.case_id.isin(tc)].copy(); t["pred"]=clf.predict(t[FEATURES]); t["ok"]=t.pred==t.label
a=t.groupby(["scenario_id","label"]).agg(rows=("ok","size"),acc=("ok","mean"),top=("pred",lambda x:x.value_counts().index[0])).round(2)
print(a[a.acc<0.9].to_string())
# equal-weight accuracy: mean of per-scenario accuracy
print("mean per-scenario accuracy (each scenario type counts once): %.3f"%t.groupby("scenario_id").ok.mean().mean())
thr=t[t.label=="threat"]; print("threat recall, mean per scenario: %.3f"%thr.groupby("scenario_id").ok.mean().mean())
