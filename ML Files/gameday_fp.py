"""False alarms on a simulated game day.

Step 1: replay every benign case the model never trained on, second by second, with the
        hold-steady rule, and record per track whether the displayed call ever became:
          - threat               (false threat alarm; for a careless hobbyist: over-escalation)
          - borderline/careless  (unneeded operator alert, for authorized or non-drone tracks)
Step 2: multiply each per-encounter probability by how often that kind of traffic shows up on a
        game day (low / typical / high assumptions) to get expected alarms per game.
Usage: python gameday_fp.py            -> gameday_tracks.csv and a printed report
"""
import gzip, pickle, sys, numpy as np, pandas as pd
from build_demo import run_case, unseen_ids
from simulate import HELD, RANK

CAT = {**{s: "Approved drones" for s in (1, 2, 3, 4, 6, 68, 69, 70, 71, 72, 73)}, 5: "Approved light show",
       7: "Crewed aircraft", 8: "Military flyover", 9: "Birds", 10: "Birds", 11: "Birds and bats at night (lights)",
       12: "Balloons, kites, fireworks", 13: "Balloons, kites, fireworks", 14: "Balloons, kites, fireworks",
       15: "Phone hotspot / insect at lens", 16: "Phone hotspot / insect at lens",
       **{s: "Hobbyist (careless, unauthorized)" for s in (17, 18, 19, 20, 21, 22, 23, 25, 36)}}
# encounters per game inside sensor coverage, TFR window (1 h before to 1 h after, about 5.5 h)
RATES = {  # (low, typical, high)
    "Hobbyist (careless, unauthorized)": (5, 10, 25),      # NFL: 2,845 incursions in the 2023 season, about 10 per game
    "Birds": (20, 60, 200),                                # assumption: flocks or singles crossing near the stadium
    "Birds and bats at night (lights)": (0, 10, 40),       # assumption: night games only
    "Balloons, kites, fireworks": (1, 3, 8),               # assumption
    "Phone hotspot / insect at lens": (5, 20, 60),         # assumption: 100,000 phones in the stands
    "Crewed aircraft": (4, 10, 20),                        # assumption: Easterwood traffic, medical and TV helicopters
    "Military flyover": (0, 1, 1),
    "Approved drones": (2, 6, 15),                         # assumption: broadcast, police, delivery corridors, research
    "Approved light show": (0, 1, 1),
}

if __name__ == "__main__":
    clf = pickle.load(gzip.open("model.pkl.gz", "rb")); clf.models[0].n_jobs = 1
    import os; pool = pd.read_csv(os.environ.get("POOL_FILE", "gameday_cases.csv"))      # fresh benign flights (make_gameday_cases.py), never seen in training
    if len(sys.argv) > 1: pool = pool.sample(int(sys.argv[1]), random_state=0)
    recs = []
    for n, (_, r) in enumerate(pool.iterrows()):
        row = r.to_dict(); tracks, st = run_case(row, clf)
        for j, tr in enumerate(tracks):
            truth = {tr["truth"][k] for k in range(len(tr["truth"]))}
            lab = row["expected_label"] if not tr["bg"] else ["authorized", "non_drone", "unauthorized_careless", "borderline", "threat"][tr["truth"][0]]
            shown = [p for p in tr["predg"] if p >= 0]; shown0 = [p for p in tr["pred"] if p >= 0]     # predg: after the alert rule; pred: model alone
            if not shown: continue
            th = sum(1 for p in shown if p == 4); th0 = sum(1 for p in shown0 if p == 4)
            alert = sum(1 for p in shown if p in (2, 3))
            notified = sum(1 for v in tr["lvl"] if v >= 2); watch = sum(1 for v in tr["lvl"] if v == 1)
            first_th = next((k for k, p in enumerate(tr["predg"]) if p == 4), None)
            recs.append(dict(case_id=row["case_id"], scenario_id=int(row["scenario_id"]), cat=CAT[int(row["scenario_id"])], bg=tr["bg"], label=lab,
                             kind=tr["kind"], seconds=len(shown), threat_s=th, alert_s=alert, ever_threat=int(th > 0), ever_alert=int(alert > 0), ever_threat_nogate=int(th0 > 0), notified_s=notified, watch_s=watch, ever_notified=int(notified > 0),
                             first_threat_t=first_th, degraded=str(row["degraded_sensor"]), lighting=row["lighting"], weather=row["weather"]))
        if n % 100 == 0: print(n, "of", len(pool), file=sys.stderr)
    df = pd.DataFrame(recs); df.to_csv(os.environ.get("OUT_FILE", "gameday_tracks.csv"), index=False)
