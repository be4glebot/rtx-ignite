"""Build the demo page.

For a handful of cases the model never trained on: run the simulator, ask the trained
model (model.pkl.gz) for its call on every track every second, apply the hold-steady rule,
and paste the results into the page. The page replays these recorded outputs.
Needs in this folder: model.pkl.gz, cuas_test_cases.csv, features.csv, demo_template.html,
three.min.js, and (optional) KyleField_demo_mesh.json.
Usage: python build_demo.py   ->  demo_data.json, cuas_demo.html
"""
import gzip, json, math, os, pickle, numpy as np, pandas as pd
import simulate as S
import alert_rules
import swarm
import tracker
from simulate import simulate_case, features_at, steady, FEATURES, PODS, BOWL_X, BOWL_Y, HELD, LABELS, RANK, CAM_RANGE_DAY, CAM_RANGE_NIGHT

PICK = [3, 5, 8, 70, 9, 12, 18, 36, 26, 24, 32, 39, 41, 42, 43, 45, 47, 80, 49, 51, 52, 53, 81, 82, 83, 69, 76]
TRY = 12            # good candidates wanted per scenario (up to TRY * 4 are looked at)
DEMO_T = 240        # demo runs are 4 minutes so slow drones can finish their scenario (training used 150 s)
MIN_SHOW = 25       # seconds of the final behaviour (hover, retreat, egress...) that must be on screen
SWARM_SIDS = {49: True, 51: True, 52: True, 53: True, 81: True, 82: True, 76: True, 83: False, 9: False, 5: False}   # True: a hostile/suspicious group should be flagged
SWARM_FLAG = {"suspicious formation", "hostile swarm (forming)", "hostile swarm", "intruder in approved group"}
PREFER_CLOSE = {9, 24, 26}   # scenarios described as crossing / heading toward the stadium: show a run that gets close
EVENT_SIDS = {26, 32, 47}   # scenarios whose behaviour changes part-way through
GOOD = 0.85         # a case is shown if the displayed call is right at least this share of the time


def r1(a):
    return [None if not np.isfinite(v) else int(round(float(v))) for v in a]


def run_case(row, clf, use_gate=True):
    """Simulate one case and return (tracks for the page, share of seconds the displayed call was right)."""
    cls = list(clf.classes_)
    _, st, objs = simulate_case(row, keep_tracks=True)
    complete = plays_out(objs, S.T_MAX)
    assoc = tracker.associate(objs, S.T_MAX, PODS, (BOWL_X, BOWL_Y), row)      # track IDs, coasting, re-association, radar handoff
    rs = np.random.default_rng(1)
    tracks, right, total, raw_right, flips_raw, flips_steady = [], 0, 0, 0, 0, 0
    right_win, events, cam_waits = 0, [], []
    cam_range = (CAM_RANGE_NIGHT if row["lighting"] == "night" else CAM_RANGE_DAY) * (0.5 if row["weather"] == "rain" else 1.0)
    for i, o in enumerate(objs):
        if o["first"] < 0: continue
        pres = np.where(o["present"])[0]
        feats, ts, fdict = [], [], []
        first0 = o["first"]
        for t in range(o["first"] + 3, min(int(pres[-1]), S.T_MAX - 1) + 1):
            seg = assoc["seg_start"][i][t]                 # the model only knows this track's own history
            if seg is None or t - seg < 3: continue
            o["first"] = seg
            f = features_at(row, objs, i, t, rs)
            o["first"] = first0
            if f is not None: feats.append([f[k] for k in FEATURES]); fdict.append(f); ts.append(t)
        raw = [None] * S.T_MAX; conf = [0] * S.T_MAX; ttb = [None] * S.T_MAX; P = None
        if feats:
            X = pd.DataFrame(feats, columns=FEATURES); P = clf.predict_proba(X)
            raw0 = [None] * S.T_MAX; latched = False; conf_at = {}
            for k, t in enumerate(ts):
                raw0[t] = cls[int(np.argmax(P[k]))]
                raw[t] = alert_rules.gate(raw0[t], fdict[k], latched) if use_gate else raw0[t]
                latched = alert_rules.update_latch(latched, fdict[k]); conf_at[t] = latched
                v = X.time_to_bowl_s.iloc[k]; ttb[t] = None if v >= 999 else int(round(v))
        ids = assoc["did"][i]
        shown = steady_by_track(raw, ids)
        shown0 = steady_by_track(raw0, ids) if feats else [None] * S.T_MAX       # what it would show without the alert rule
        fd_at = dict(zip(ts, fdict)); lvl = [0] * S.T_MAX; lastf = None
        if not feats: conf_at = {}
        for t in range(S.T_MAX):
            lastf = fd_at.get(t, lastf)
            if shown[t] is not None and lastf is not None: lvl[t] = alert_rules.LEVELS.index(alert_rules.level(shown[t], lastf, conf_at.get(t, any(v for k2, v in conf_at.items() if k2 <= t))))
        for k, t in enumerate(ts):
            if shown0[t] is not None: conf[t] = int(round(100 * P[k][cls.index(shown0[t])]))
        truth = []
        for t in range(S.T_MAX):
            lab = o["label"] if o["bg"] else row["expected_label"]
            if not o["bg"] and o.get("event_t") is not None and o.get("before") and t < o["event_t"]: lab = o["before"]
            truth.append(lab)
        if not o["bg"]:
            # behaviour change (e.g. hijack): the label steps up in concern at ev
            ev = next((k for k in range(1, S.T_MAX) if RANK[truth[k]] > RANK[truth[k - 1]]), None)
            for t in ts:
                total += 1; right += shown0[t] == truth[t]; raw_right += raw0[t] == truth[t]
                # transition window: from 15 s before to 5 s after the change, an early "borderline" or either side's label counts as right
                ok_win = shown0[t] == truth[t] or (ev is not None and ev - 15 <= t <= ev + 5 and shown0[t] in (truth[ev - 1], "borderline", truth[ev]))
                right_win += ok_win
            if ev is not None and ev < S.T_MAX - 5:
                alert = next((t for t in ts if shown0[t] is not None and RANK[shown0[t]] >= RANK["borderline"] and t >= ev - 30), None)
                thr = next((t for t in ts if shown0[t] == truth[ev] and t >= ev - 30), None) if truth[ev] == "threat" else None
                events.append(dict(sid=int(row["scenario_id"]), ev=ev, alert=None if alert is None else alert - ev, threat=None if thr is None else thr - ev, to=truth[ev]))
            flips_raw += sum(1 for a, b in zip(ts, ts[1:]) if raw0[a] != raw0[b]); flips_steady += sum(1 for a, b in zip(ts, ts[1:]) if shown0[a] != shown0[b])
        # how long a drone waited for a camera once it was tracked and within camera range
        if o["kind"] in ("multirotor_drone", "fixed_wing_drone"):
            inr = np.where(o["present"] & (np.arange(S.T_MAX) >= o["first"]) & (np.nan_to_num(o["cam_dist_any"], nan=1e9) <= cam_range))[0]
            if len(inr):
                lk = np.where(o["cam_lock"] > 0)[0]; lk = lk[lk >= inr[0]]
                cam_waits.append((int(row["n_objects"]) > 1, (lk[0] - inr[0]) if len(lk) else None))
        # per-second facts the swarm module needs (from the same features the model saw)
        onman = [0] * S.T_MAX; confd = [0] * S.T_MAX; radio = [0] * S.T_MAX; confn = [0] * S.T_MAX; cbird = [0] * S.T_MAX; lat = False
        for t_, f_ in zip(ts, fdict):
            c_ = alert_rules.confirmed(f_); lat = alert_rules.update_latch(lat, f_)
            onman[t_] = int(f_["on_manifest"]); confd[t_] = int(lat); radio[t_] = int(f_["rf_frac"] > 0.1 or f_["rid_seen"])
            confn[t_] = int(c_); cbird[t_] = int(f_["cam_bird"] and f_["cam_looks"] >= 2)
        for t_ in range(1, S.T_MAX):          # carry forward between feature seconds
            if t_ not in fd_at: onman[t_] = onman[t_ - 1]; confd[t_] = confd[t_ - 1]; radio[t_] = radio[t_ - 1]; confn[t_] = confn[t_ - 1]; cbird[t_] = cbird[t_ - 1]
        # camera identification per second (1 drone, 2 bird, 3 other) and "needs human review"
        cid = [0] * S.T_MAX; rev = [0] * S.T_MAX
        for t_, f_ in zip(ts, fdict):
            cid[t_] = 1 if f_["cam_drone"] else 2 if f_["cam_bird"] else 3 if f_["cam_other"] else 0
        for t_ in range(1, S.T_MAX):
            if t_ not in fd_at: cid[t_] = cid[t_ - 1]
        for k_, t_ in enumerate(ts):
            f_ = fdict[k_]; lab_ = shown0[t_]
            unsure = lab_ in ("unauthorized_careless", "borderline", "threat") and conf[t_] < REVIEW_CONF
            conflict = f_["rid_seen"] and f_["rid_pos_err_m"] > 100
            rev[t_] = int(unsure or conflict or assoc["doubt"][i][t_])
        for t_ in range(1, S.T_MAX):
            if t_ not in fd_at and assoc["did"][i][t_]: rev[t_] = rev[t_ - 1]
        det = (o["rad_det"] * 1 + o["rf_det"] * 2 + o["rid_det"] * 4 + o["cam_det"] * 8).astype(int).tolist()
        tracks.append(dict(bg=int(o["bg"]), kind=o["kind"], first=int(o["first"]),
                           tx=r1(o["truth"][:, 0]), ty=r1(o["truth"][:, 1]), tz=r1(o["truth"][:, 2]),
                           fx=r1(o["fused"][:, 0]), fy=r1(o["fused"][:, 1]), det=det,
                           pred=[-1 if p is None else LABELS.index(p) for p in shown0], predg=[-1 if p is None else LABELS.index(p) for p in shown], lvl=lvl, onman=onman, confd=confd, confn=confn, cbird=cbird, radio=radio, conf=conf, ttb=ttb,
                           truth=[LABELS.index(x) for x in truth], cam=o["cam_class"], lock=o["cam_lock"].astype(int).tolist(),
                           att=o["att_obs"].astype(int).tolist(), attc=[int(round(100 * v)) for v in o["att_conf"]],
                           did=assoc["did"][i], st=assoc["state"][i], unc=assoc["unc"][i], ex=r1(np.array([np.nan if v is None else v for v in assoc["ex"][i]])),
                           ey=r1(np.array([np.nan if v is None else v for v in assoc["ey"][i]])), rad=assoc["radar"][i], asg=o["cam_asg"].astype(int).tolist(), cid=cid, rev=rev))
    stats = dict(cams=(objs[0]["cams_ok"] if objs else []), cam_range=(objs[0]["cam_range"] if objs else 0), assoc=assoc["stats"], track_events=assoc["events"], complete=complete, right_win=right_win, events=events, cam_waits=cam_waits, total=total, right=right, raw_right=raw_right, flips_raw=flips_raw, flips_steady=flips_steady, detected=st["detected"])
    return tracks, stats


REVIEW_CONF = 60    # a concern call the model is less than 60% sure of is flagged for human review


def steady_by_track(raw, ids):
    """Hold-steady rule, restarted whenever the track ID changes (a new track has no earlier call to hold)."""
    out = [None] * len(raw); start = 0
    for t in range(1, len(raw) + 1):
        if t == len(raw) or ids[t] != ids[start]:
            seg = steady(raw[start:t])
            out[start:t] = seg if ids[start] else [None] * (t - start)
            start = t
    return out


def plays_out(objs, T):
    """True if every foreground object finishes its scenario inside the window: all plan segments start,
    a final approach into the stadium actually arrives, and the closing behaviour is on screen for MIN_SHOW s."""
    for o in objs:
        if o["bg"]: continue
        plan, st = o["plan"], o["seg_t"]
        if len(st) < len(plan) or st[-1] >= T: return False
        kind, dest = plan[-1]
        ok = np.where(np.isfinite(o["truth"][:, 0]))[0]
        if kind == "goto" and np.hypot(dest[0], dest[1]) < 150:
            if not len(ok) or ok[-1] >= T - 3: return False          # must reach the stadium before the run ends
        elif st[-1] > T - MIN_SHOW and len(plan) > 1:
            return False
        if o.get("event_t") is not None and o["event_t"] > T - MIN_SHOW: return False
    return True


def unseen_ids():
    """Cases the model did not train on: the 20% test split used in train.py."""
    ids = np.array(pd.read_csv("features.csv", usecols=["case_id"]).case_id.unique()); np.random.default_rng(7).shuffle(ids)
    return set(ids[:int(len(ids) * 0.2)])


SENSOR_KEYS = ["", "no_radar", "no_rf", "no_cam", "no_radar+no_rf", "no_radar+no_cam", "no_rf+no_cam"]   # every on/off combination except all off
CONDS = {"fog": "Fog", "cameras_out_2": "Two cameras down", "rf_interference": "Heavy RF interference", "crowded": "Several objects arriving together"}
POOL_N = int(os.environ.get("POOL_N", 25))     # random runs per challenge condition
KEEP = ("bg", "kind", "first", "tx", "ty", "tz", "fx", "fy", "det", "pred", "lvl", "conf", "ttb", "cam", "lock", "att", "attc",
        "slvl", "grp", "did", "st", "unc", "ex", "ey", "rad", "asg", "cid", "rev")


def pack(tracks, st):
    """Swarm module on the run, then keep only what the page draws."""
    gby, slvl, gof = swarm.analyze(tracks, S.T_MAX)
    out = []
    for i, tr in enumerate(tracks):
        tr["slvl"] = slvl[i]; tr["grp"] = [g or 0 for g in gof[i]]
        out.append({k: tr[k] for k in KEEP})
    groups = [[[g["id"], g["status"], sorted(g["members"]), g["size"], g["vel_spread"], g["spacing_cv"], g["aim_spread"], g["eta_spread"],
                int(g["simultaneous_entry"]), g["persistence"], g["radio_silent"], g["on_manifest"], g["confirmed"], "/".join(g["kinds"]), g["split_merge"]]
               for g in gby[t]] for t in range(S.T_MAX)]
    return dict(tracks=out, groups=groups, tev=st["track_events"], cams=st["cams"], cam_range=round(st["cam_range"]), assoc={k: int(v) for k, v in st["assoc"].items()})


def meta(row):
    return dict(scenario_id=int(row["scenario_id"]), name=row["scenario_name"], case_id=str(row["case_id"]), expected=row["expected_label"],
                held=int(int(row["scenario_id"]) in HELD), phase=str(row["phase"]), lighting=str(row["lighting"]), weather=str(row["weather"]),
                n_true=int(row["n_objects"]), rf_link=str(row["rf_link"]), remote_id=str(row["remote_id"]), on_list=str(row["on_approved_list"]))


def challenge_rows(n_per):
    """Fresh random cases for the challenge pool: random scenario type, new seed, one condition each. Not hand-picked."""
    src = open("gen_cases.py").read(); src = src[:src.index("def add(r):")].replace("random.Random(20261116)", "random.Random(31337)")
    ns = {}; exec(src, ns)
    types = [s for s in ns["S"] if s[0] not in (1, 2)]          # skip far-off delivery corridors (nothing happens near the stadium)
    rows = []
    for cond in CONDS:
        for k in range(n_per):
            r = ns["make"](ns["R"].choice(types)); r["degraded_sensor"] = cond
            if cond == "crowded": r["concurrent_event"] = "several_arrivals"
            r["case_id"] = "CH-%s-%02d" % (cond, k); rows.append(r)
    return rows


if __name__ == "__main__":
    S.T_MAX = DEMO_T; S.DEMO_MODE = True
    clf = pickle.load(gzip.open("model.pkl.gz", "rb")); clf.models[0].n_jobs = 1
    cases = pd.read_csv("cuas_test_cases.csv"); UNSEEN = unseen_ids()
    if os.path.exists("swarm_cases.csv"):     # swarm-study cases (new seed, never trained on), incl. scenarios 81-83
        sw = pd.read_csv("swarm_cases.csv"); sw = sw[sw.scenario_id.isin([81, 82, 83])]
        cases = pd.concat([cases, sw], ignore_index=True); UNSEEN |= set(sw.case_id)
    out = []
    if os.environ.get("PICK_ONLY"): PICK = [int(x) for x in os.environ["PICK_ONLY"].split(",")]
    for sid in PICK:
        cand = cases[(cases.scenario_id == sid) & (cases.degraded_sensor.astype(str) == "none") & (cases.case_id.isin(UNSEEN) | (sid in HELD))].head(TRY * 4)
        good, tried, keep = 0, 0, []
        for _, row in cand.iterrows():
            row = row.to_dict(); tracks, st = run_case(row, clf)
            if st["detected"] < 1 or st["total"] == 0 or not st["complete"]: continue
            tl = [tr["truth"] for tr in tracks if not tr["bg"]]
            if sid in EVENT_SIDS and not any(len(set(x)) > 1 for x in tl): continue   # show a run where the change actually happens
            sc = st["right_win"] / st["total"]; tried += 1; good += sc >= GOOD
            mind = min(min((math.hypot(x, y) for x, y in zip(t["tx"], t["ty"]) if x is not None), default=1e9) for t in tracks if not t["bg"])
            ok = True
            if sid in SWARM_SIDS:     # swarm scenarios: the run must show the right GROUP outcome
                gby, _, _ = swarm.analyze(tracks, S.T_MAX); fg = {i for i, t in enumerate(tracks) if not t["bg"]}
                flagged = any(g["status"] in SWARM_FLAG and len(g["members"] & fg) >= (1 if sid == 76 else 2) for gs in gby for g in gs)
                ok = flagged if SWARM_SIDS[sid] else not flagged
            keep.append((sc, row, tracks, mind, ok, st))
            if tried >= (TRY * 3 if sid in PREFER_CLOSE else TRY): break
        if not keep: print("no detected case for", sid); continue
        pool = [k for k in keep if k[4]] or keep
        pool = [k for k in pool if k[0] >= GOOD] or pool
        sc, row, tracks, _, _, st = min(pool, key=lambda k: k[3]) if sid in PREFER_CLOSE else max(pool, key=lambda k: k[0])
        runs = {"": pack(tracks, st)}
        for key in SENSOR_KEYS[1:]:          # same flights, sensors switched off
            r2 = dict(row); r2["degraded_sensor"] = key
            tr2, st2 = run_case(r2, clf); runs[key] = pack(tr2, st2)
        out.append(dict(meta(row), runs=runs))
        print("%2d %-58s right %3.0f%% (transition window) | %d of %d candidates good" % (sid, row["scenario_name"][:58], 100 * sc, good, tried), flush=True)
    chal = []
    for row in challenge_rows(POOL_N):
        tr, st = run_case(row, clf)
        if not tr: continue
        chal.append(dict(meta(row), cond=row["degraded_sensor"], run=pack(tr, st)))
    print("challenge pool:", len(chal), "runs", flush=True)
    PODNAME = {"NE": "NE corner", "NW": "NW corner", "SE": "SE corner", "SW": "SW corner", "CCG": "Central garage", "WCG": "West garage"}
    bld = json.load(open("buildings_demo.json"))
    data = dict(labels=LABELS, pods=PODS, bowl=[BOWL_X, BOWL_Y], T=S.T_MAX, cases=out, pool=chal, conds=CONDS,
                radars=list(S.RADARS), podz={k: S.SITE_Z[k] for k in PODS}, podname=PODNAME, bld=bld)
    js = json.dumps(data, separators=(",", ":"))
    import base64
    z = base64.b64encode(gzip.compress(js.encode(), 9)).decode()
    print("data: %.1f MB raw, %.1f MB compressed" % (len(js) / 1e6, len(z) / 1e6))
    html = open("demo_template.html").read().replace("/*__DATAZ__*/", z).replace("/*__THREE__*/", open("three.min.js").read())
    if os.path.exists("KyleField_demo_mesh.json"):   # stadium + pods exported from FreeCAD; without it the page draws a simple stand-in
        html = html.replace("/*__MESH__*/null", open("KyleField_demo_mesh.json").read())
    open("cuas_demo.html", "w").write(html)
    print("page size %.1f MB" % (len(html) / 1e6))
