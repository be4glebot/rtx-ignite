"""Operator burden: benign notifications per game, split into brief ones (camera resolves them within 15 s)
and lingering ones (still flagged after 15 s: a human has to look)."""
import pandas as pd
from gameday_fp import RATES
d = pd.read_csv("gameday_tracks.csv"); d = d[d.label.isin(["authorized", "non_drone"])]
e = d.groupby(["case_id", "cat"]).notified_s.max().reset_index()
tab = e.groupby("cat").agg(n=("notified_s", "size"), brief=("notified_s", lambda x: ((x > 0) & (x <= 15)).mean()), linger=("notified_s", lambda x: (x > 15).mean()))
tot = [[0, 0], [0, 0], [0, 0]]
for cat, r in tab.iterrows():
    for i, rate in enumerate(RATES[cat]): tot[i][0] += rate * r.brief; tot[i][1] += rate * r.linger
print(tab.round(3).to_string())
print("benign notifications per game (low / typical / high): brief %.1f / %.1f / %.1f, lingering >15 s %.1f / %.1f / %.1f" % (tot[0][0], tot[1][0], tot[2][0], tot[0][1], tot[1][1], tot[2][1]))
