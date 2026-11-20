"""Train and evaluate the behaviour model on simulated tracks."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from simulate import FEATURES, HELD, LABELS
df=pd.read_csv("features.csv"); sd=pd.read_csv("case_stats.csv")
rng=np.random.default_rng(7)
cases=np.array(df.case_id.unique()); rng.shuffle(cases); test_cases=set(cases[:int(len(cases)*0.2)])
held=df.scenario_id.isin(HELD)
train=df[~held & ~df.case_id.isin(test_cases)]; test=df[~held & df.case_id.isin(test_cases)]; ho=df[held]
# weights: the extra weak-point cases must not drown out the other scenario types, and missing a threat costs more than over-calling one
_n=train.groupby("scenario_id").label.transform("size"); _w=1/np.sqrt(_n)
_cw=train.label.map(1/(pd.Series(_w.values).groupby(train.label.values).sum())); _w=_w*_cw.values*train.label.map({"threat":1.6}).fillna(1.0).values
import os
from sklearn.ensemble import HistGradientBoostingClassifier
MODEL=os.environ.get("MODEL","avg")      # avg (default) = random forest + gradient boosting averaged; rf or gb = one alone
def make(kind):
    if kind=="gb": return HistGradientBoostingClassifier(max_iter=500,learning_rate=0.06,max_leaf_nodes=48,min_samples_leaf=20,l2_regularization=1.0,random_state=0)
    return RandomForestClassifier(n_estimators=300,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=0)
parts=[make(k).fit(train[FEATURES],train.label,sample_weight=_w/_w.mean()) for k in (["rf","gb"] if MODEL=="avg" else [MODEL])]
if MODEL=="avg":
    from cuas_model import AvgModel; clf=AvgModel(parts)
else: clf=parts[0]
print("model:",MODEL,[type(p).__name__ for p in parts])
def rep(name,d):
    p=clf.predict(d[FEATURES]); acc=(p==d.label).mean()
    print("\n== %s: %d rows, %d cases, accuracy %.3f"%(name,len(d),d.case_id.nunique(),acc))
    print(pd.DataFrame(confusion_matrix(d.label,p,labels=LABELS),index=LABELS,columns=[l[:6] for l in LABELS]).to_string())
    return p
pt=rep("TEST (seen scenario types, unseen cases)",test)
ph=rep("HELD-OUT scenario types",ho)
hod=ho.assign(pred=ph,ok=(ph==ho.label.values)); print(hod.groupby(["scenario_id","label"]).agg(rows=("ok","size"),acc=("ok","mean"),top_pred=("pred",lambda x:x.value_counts().index[0])).round(2).to_string())
if hasattr(parts[0],"feature_importances_"):
    imp=pd.Series(parts[0].feature_importances_,index=FEATURES).sort_values(ascending=False); print("\nTop features:\n",imp.head(12).round(3).to_string())
# threat recall by checkpoint (how early)
t=test.assign(pred=pt); thr=t[t.label=="threat"]
print("\nThreat recall by seconds since first detection:\n",thr.assign(hit=thr.pred=="threat").groupby("checkpoint_s").hit.mean().round(3).to_string())
fa=t[t.label.isin(["authorized","non_drone"])]; print("False threat alarms on authorized/non-drone rows: %.4f"%((fa.pred=="threat").mean()))
# anomaly scorer trained on authorized traffic only
from sklearn.metrics import roc_auc_score
unc=lambda d: 1-clf.predict_proba(d[FEATURES]).max(axis=1)
t["unfamiliar"]=unc(t); hod["unfamiliar"]=unc(ho)
print("\nUnfamiliarity score (1 - model confidence): seen types mean %.3f, held-out types mean %.3f, AUC held-out vs seen %.3f"%(t.unfamiliar.mean(),hod.unfamiliar.mean(),roc_auc_score([0]*len(t)+[1]*len(hod),list(t.unfamiliar)+list(hod.unfamiliar))))
print("by held-out scenario:",hod.groupby("scenario_id").unfamiliar.mean().round(2).to_dict())
print("Unfamiliarity when the model is wrong %.3f vs right %.3f (test)"%(t[t.pred!=t.label].unfamiliar.mean(),t[t.pred==t.label].unfamiliar.mean()))
m=t[t.on_manifest==1]; print("Manifest deviation (m), median by label, for tracks on the manifest:",m.groupby("label").manifest_dev_m.median().round(0).to_dict())
print("\nDetection stats: undetected cases by label:",sd[sd.detected==0].groupby("label").size().to_dict())
w=sd[(sd.label=="threat")&sd.warning_s.notna()]; print("Threat warning time, drones starting OUTSIDE: median %.0f s, 10th pct %.0f s, n=%d"%(w.warning_s.median(),w.warning_s.quantile(0.1),len(w)))
ins=sd[(sd.starts_inside==1)&(sd.label=="threat")]; print("Threats starting INSIDE: %d cases, detected %d, median detection delay %.0f s"%(len(ins),(ins.detected>0).sum(),ins.detect_delay_s.median()))
print("undetected threat scenarios:",sd[(sd.detected==0)&(sd.label=="threat")].scenario_id.value_counts().to_dict())
print("undetected borderline:",sd[(sd.detected==0)&(sd.label=="borderline")].scenario_id.value_counts().to_dict()); print("undetected authorized:",sd[(sd.detected==0)&(sd.label=="authorized")].scenario_id.value_counts().to_dict())

import pickle, gzip; pickle.dump(clf,gzip.open(os.environ.get("OUT","model.pkl.gz"),"wb",compresslevel=6))   # compressed: the forest is ~90 MB otherwise
