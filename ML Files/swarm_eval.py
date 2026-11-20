"""Swarm study: replay the swarm cases (4-minute runs) through sensors, the per-drone model, the alert
rule and the swarm module, and score at the GROUP level.
Usage: python swarm_eval.py [shard] [nshards]   ->  swarm_eval_<shard>.csv ;  python swarm_eval.py report"""
import gzip, math, os, pickle, sys, pandas as pd
import simulate as S
S.T_MAX = 240; S.DEMO_MODE = True
from build_demo import run_case
import swarm

HOSTILE = {51, 52, 53, 75, 81, 82, 76, 54, 55, 56, 49}
FLAG = {"suspicious formation", "hostile swarm (forming)", "hostile swarm", "intruder in approved group"}


def simulate_all(shard, ns):
    """Run sensors + model + alert rule once per case and cache the tracks, so swarm rules can be re-scored quickly."""
    import os
    clf = pickle.load(gzip.open("model.pkl.gz", "rb")); clf.models[0].n_jobs = 1
    cases = pd.read_csv("swarm_cases.csv").iloc[shard::ns]
    os.makedirs("swarm_cache_v9", exist_ok=True)
    for n, (_, r) in enumerate(cases.iterrows()):
        fn = "swarm_cache_v9/%s.pkl.gz" % r.case_id
        if os.path.exists(fn): continue
        tracks, st = run_case(r.to_dict(), clf)
        pickle.dump(tracks, gzip.open(fn, "wb", compresslevel=3))
        if n % 50 == 0: print(shard, n, "of", len(cases), file=sys.stderr, flush=True)


def score(row, clf=None):
    T = S.T_MAX
    tracks = pickle.load(gzip.open("swarm_cache_v9/%s.pkl.gz" % row["case_id"], "rb"))
    groups, level, gof = swarm.analyze(tracks, T)
    fg = [i for i, tr in enumerate(tracks) if not tr["bg"]]
    out = dict(case_id=row["case_id"], sid=int(row["scenario_id"]), n_fg=len(fg), n_tracks=len(tracks), hostile=int(row["scenario_id"]) in HOSTILE)
    if not fg: return out
    first = min(tracks[i]["first"] for i in fg)
    dist = lambda i, k: None if tracks[i]["tx"][k] is None else math.hypot(tracks[i]["tx"][k], tracks[i]["ty"][k])
    reach = next((k for k in range(T) if any(dist(i, k) is not None and dist(i, k) < 120 and k >= tracks[i]["first"] for i in fg)), None)
    need = 1 if int(row["scenario_id"]) == 76 else 2
    flag_t = alarm_t = None; ids = set(); cover = 0.0; recognized = None; statuses = set()
    for t in range(T):
        for g in groups[t]:
            statuses.add(g["status"])
            if recognized is None and len(g["members"]) >= 2: recognized = t
            k = len(g["members"] & set(fg))
            if g["status"] in FLAG and k >= need:
                if flag_t is None:
                    flag_t = t
                    det = [i for i in fg if tracks[i]["first"] <= t]
                    cover = k / max(len(det), 1)
                ids.add(g["id"])
            if g["status"] == "hostile swarm" and k >= need and alarm_t is None: alarm_t = t
    ind_alert = next((t for t in range(T) if any(tracks[i]["lvl"][t] >= 2 for i in fg)), None)
    ind_alarm = next((t for t in range(T) if any(tracks[i]["lvl"][t] >= 3 for i in fg)), None)
    any_flag = any(g["status"] in FLAG for t in range(T) for g in groups[t])
    any_alarm = any(g["status"] == "hostile swarm" for t in range(T) for g in groups[t])
    out.update(first=first, reach=reach, flag_t=flag_t, alarm_t=alarm_t, ids=len(ids), cover=round(cover, 2), recognized=recognized,
               ind_alert=ind_alert, ind_alarm=ind_alarm, any_flag=int(any_flag), any_alarm=int(any_alarm), statuses="|".join(sorted(statuses)))
    return out


def report():
    import glob
    d = pd.concat([pd.read_csv(f) for f in glob.glob("swarm_eval_*.csv")])
    names = {**pd.read_csv("swarm_cases.csv").drop_duplicates("scenario_id").set_index("scenario_id").scenario_name.to_dict()}
    h = d[d.hostile]
    rows = []
    for sid, g in h.groupby("sid"):
        fl = g.flag_t.notna(); before = fl & (g.reach.isna() | (g.flag_t <= g.reach))
        rows.append(dict(scenario=names[sid][:58], cases=len(g), swarm_flagged=fl.mean(), flagged_before_arrival=before.mean(),
                         sec_after_first_detection=(g.flag_t - g.first)[fl].median(), warning_s=(g.reach - g.flag_t)[fl & g.reach.notna()].median(),
                         members_in_group=g.cover[fl].median(), group_ids=g.ids[fl].median(),
                         individual_alert_ever=g.ind_alert.notna().mean(), individual_alarm_ever=g.ind_alarm.notna().mean()))
    pd.set_option("display.width", 250); print("HOSTILE GROUPS\n" + pd.DataFrame(rows).round(2).to_string(index=False))
    b = d[~d.hostile]; rows = []
    for sid, g in b.groupby("sid"):
        nm = names.get(sid, "")
        if sid == 17: nm = "Ten or more hobbyists at once (independent)"
        rows.append(dict(scenario=nm[:58], cases=len(g), coordination_recognized=g.recognized.notna().mean(), false_swarm_alert=g.any_flag.mean(), false_swarm_alarm=g.any_alarm.mean(),
                         statuses=", ".join(sorted(set("|".join(g.statuses.dropna()).split("|")) - {""}))[:70]))
    print("\nBENIGN GROUPS\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report": report(); sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "simulate": simulate_all(int(sys.argv[2]), int(sys.argv[3])); sys.exit()
    import glob
    for f in glob.glob("swarm_eval_*.csv"): os.remove(f) if False else None
    cases = pd.read_csv("swarm_cases.csv")
    recs = [score(r.to_dict()) for _, r in cases.iterrows()]
    pd.DataFrame(recs).to_csv("swarm_eval_0.csv", index=False)
    report()
