import sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
import simulate as S
def work(chunk):
    rows, stats = [], []
    for r in chunk:
        rr, st, _ = S.simulate_case(r)
        rows += rr; st.update(scenario_id=r["scenario_id"], label=r["expected_label"], group=r["group"]); stats.append(st)
    return rows, stats
if __name__ == "__main__":
    t = time.time()
    cases = pd.read_csv("cuas_test_cases.csv").to_dict("records")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cases)
    cases = cases[:n]
    chunks = [cases[i::16] for i in range(16)]
    with Pool(2) as p: res = p.map(work, chunks)
    rows = [x for a, _ in res for x in a]; stats = [x for _, b in res for x in b]
    df = pd.DataFrame(rows); sd = pd.DataFrame(stats)
    order = {c: i for i, c in enumerate(pd.read_csv("cuas_test_cases.csv").case_id)}
    df = df.sort_values(["case_id", "obj", "checkpoint_s"], key=lambda c: c.map(order) if c.name == "case_id" else c).reset_index(drop=True)
    sd = sd.sort_values("case_id", key=lambda c: c.map(order)).reset_index(drop=True)
    if n == len(order):
        df.to_csv("features.csv", index=False); sd.to_csv("case_stats.csv", index=False)
    else:
        print(sd.groupby(["label","size"]).agg(n=("case_id","size"), det=("detected", lambda x:(x>0).mean()), warn=("warning_s","median")))
    print("cases", len(sd), "rows", len(df), "time %.0fs" % (time.time() - t))
