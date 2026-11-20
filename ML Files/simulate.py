"""Counter-UAS track simulator for the Kyle Field concept.

Turns one row of cuas_test_cases.csv into:
  1. ground truth  - where every object really is, once per second
  2. sensor view   - what the 2 radars, 6 RF sensors and 6 cameras would report (six sites: 4 stadium corners,
                     Central Campus Garage and West Campus Garage; line of sight uses the real stadium + campus buildings, geo.py)
  3. fused track   - one best estimate per object per second, compared with the manifest
and then into features the model learns from.

All sensor performance numbers below are ASSUMPTIONS (the challenge prompt allows
assumed performance levels). Coordinates: metres, origin at midfield, x across the field (+x = west side),
y along the field (+y = north end).
"""
import math, json, os
import numpy as np
import pandas as pd
import geo

DT = 1.0
T_MAX = 150
DEMO_MODE = False            # set True by build_demo.py (longer runs, no mid-air vanishing)
MAX_OBJ = 12                 # objects simulated per case (large groups are capped)
PODS = {"NE": (100, 88), "NW": (-100, 88), "SE": (100, -88), "SW": (-100, -88),        # stadium corner roofs (CAD model): camera + RF
        "CCG": tuple(round(v) for v in geo.GARAGES["Central Campus Garage"]["sim_spot"]),    # garage roofs: radar + camera + RF
        "WCG": tuple(round(v) for v in geo.GARAGES["West Campus Garage"]["sim_spot"])}
RADARS = ("CCG", "WCG")
GARAGE_ROOF = 23.9           # both garages: 7 levels (confirmed); sensors on a short mast above the top deck
SITE_Z = {k: {"cam": 39.0, "rf": 40.0, "radar": None} for k in ("NE", "NW", "SE", "SW")}
SITE_Z.update({k: {"cam": GARAGE_ROOF + 3.0, "rf": GARAGE_ROOF + 3.4, "radar": GARAGE_ROOF + 4.2} for k in ("CCG", "WCG")})   # garage kit CAD: camera ~3.0 m, RF ~3.4 m, radar ~4.2 m above the deck
RF_BEARING_SD = math.radians(5.0)   # RF direction-finding bearing accuracy per sensor (Dedrone RF-300 datasheet: +-5 deg)
BOWL_X, BOWL_Y = 118.0, 115.0   # radar cannot see inside this footprint
CHECKPOINTS = (5, 10, 20, 30, 45, 60, 90)
HELD = [10, 22, 31, 43, 52, 70, 78]     # scenario types never shown in training (v3 set)
LABELS = ["authorized", "non_drone", "unauthorized_careless", "borderline", "threat"]
RANK = {"non_drone": 0, "authorized": 1, "unauthorized_careless": 2, "borderline": 3, "threat": 4}
UP_S, DOWN_S = 2, 8          # seconds a new call must persist before the displayed call changes


def steady(raw):
    """Hold-steady rule. raw: list of labels per second (None = no call yet).
    The displayed call moves UP in concern after UP_S consecutive seconds, DOWN only after DOWN_S."""
    out, state, cand, n = [], None, None, 0
    for p in raw:
        if p is None:
            out.append(state); continue
        if state is None:
            state = p
        elif p == state:
            cand, n = None, 0
        else:
            if p == cand: n += 1
            else: cand, n = p, 1
            if n >= (UP_S if RANK[p] > RANK[state] else DOWN_S):
                state, cand, n = p, None, 0
        out.append(state)
    return out

# ---- assumed sensor performance -------------------------------------------------
RADAR_RANGE = 5000.0
RADAR_PD = {"multirotor_drone": 0.90, "fixed_wing_drone": 0.92, "bird": 0.60, "bird_or_bat": 0.45,
            "balloon": 0.30, "pyrotechnic": 0.50, "debris": 0.30, "crewed_aircraft": 0.99,
            "military_jet": 0.99, "rf_only_ghost": 0.0, "camera_only_ghost": 0.0}
RADAR_RANGE_ACC = 0.6        # radar range accuracy (m), IRIS datasheet
RADAR_ANG_ACC = math.radians(0.7)   # radar angular accuracy (0.6-0.75 deg, IRIS datasheet): side-to-side error grows with range
STAND_H = {"ew": 52.0, "ns": 42.0}   # stadium top heights: east/west decks ~170 ft, north/south lower
RF_RANGE = 3000.0
RF_PD_QUIET, RF_PD_CROWD = 0.90, 0.75
RID_FULL_M, RID_MAX_M = 400.0, 1000.0   # Remote ID: reliably heard to ~400 m, fading to nothing by 1 km (field tests: ~100-500 m with phone apps)
RID_PD_QUIET, RID_PD_CROWD = 0.60, 0.40
# camera grades: (day, night) range on a small consumer drone (DJI Mavic / Phantom class)
CAM_GRADE_RANGE = {"standard": (1000.0, 400.0),     # Dedrone standard PTZ: ~1 km day, ~400 m night (IR)
                   "long": (2000.0, 800.0)}         # Dedrone long-range PTZ: ~2 km day; night 800 m is our assumption (same ratio)
CAM_GRADE = {"NE": "standard", "NW": "standard", "SE": "standard", "SW": "standard", "CCG": "long", "WCG": "long"}
CAM_RANGE_DAY, CAM_RANGE_NIGHT = CAM_GRADE_RANGE["standard"]   # reference range (the corner cameras)
# drone size classes. radar_m: range where radar detection has faded to zero (full rate to 75% of it).
# Anchors: IRIS 1.1 km on a 0.249 kg drone and 7.6 km on 52 kg; the classes in between sit on a log-log line through those points.
# cam: camera range multiplier vs a small drone (more pixels on target); capped well below the size ratio to stay conservative.
SIZE_CLASSES = {"micro": dict(desc="under 250 g (DJI Mini class)", mass_kg=0.249, radar_m=1100.0, cam=0.7),
                "small": dict(desc="~1 kg consumer (Mavic, Phantom, FPV)", mass_kg=1.0, radar_m=1800.0, cam=1.0),
                "medium": dict(desc="~4 kg prosumer / enterprise (Inspire, Matrice 30)", mass_kg=4.0, radar_m=3000.0, cam=1.4),
                "large": dict(desc="~10 kg heavy lift (Matrice 350, delivery, sprayer)", mass_kg=10.0, radar_m=4200.0, cam=1.8)}
MANIFEST_SIGMA = 2.5        # a deviation from the filed plan counts only beyond 2.5x the track's own position error (about 4% chance by noise alone)
RADAR_GHOST_M = 0.75 * SIZE_CLASSES["micro"]["radar_m"]   # ghost check: only inside the range where even a sub-250 g drone shows on radar
NONDRONE_RADAR_M = {"bird": 1800.0, "bird_or_bat": 1100.0, "balloon": 1500.0, "pyrotechnic": 2000.0, "debris": 1200.0,
                    "crewed_aircraft": 8000.0, "military_jet": 8000.0}
NONDRONE_CAM = {"crewed_aircraft": 4.0, "military_jet": 4.0, "balloon": 0.8, "pyrotechnic": 1.5}
CAM_ACQ_S = 3                # seconds for a PTZ camera to slew, zoom and focus on a newly assigned track
CAM_DWELL_S = 6              # minimum seconds a camera stays on a track before it can be re-tasked
CAM_SWEEP_PD = 0.08          # chance per second that an idle camera sweeping the bowl airspace picks up a given object there
import os
BIRD_MD = float(os.environ.get("BIRD_MD", 0.12))          # chance per radar return that a bird's wing beats look rotor-like (stress tests raise it)
CAM_ID_FRAC = float(os.environ.get("CAM_ID_FRAC", 0.6))   # identification range as a share of camera detection range (DRI: identify needs more pixels than detect)
CAM_ID_SCALE = float(os.environ.get("CAM_ID_SCALE", 1.0)) # camera identification accuracy multiplier (stress tests lower it)
ADSB_SHARE = 0.8             # share of crewed aircraft broadcasting ADS-B (not required in Easterwood's Class D airspace)            # camera height on the pavilion-roof mast (m)


def pol(bearing_deg, rng):
    b = math.radians(bearing_deg)
    return np.array([rng * math.sin(b), rng * math.cos(b)])


def bearing_of(p):
    return math.degrees(math.atan2(p[0], p[1])) % 360


# ---- trajectory building ----------------------------------------------------------
def run_plan(start, segs, rs):
    """Integrate a list of segments into a (T_MAX, 3) array; NaN where the object is absent."""
    out = np.full((T_MAX, 3), np.nan)
    p = np.array(start, float)
    t = 0
    seg_t = []
    for s in segs:
        seg_t.append(t)
        if t >= T_MAX:
            break
        kind = s[0]
        if kind == "wait":
            t += int(s[1])
        elif kind == "goto":
            tgt, spd = np.array(s[1], float), max(float(s[2]), 0.3)
            while t < T_MAX:
                d = tgt - p
                dist = np.linalg.norm(d)
                if dist <= spd * DT:
                    p = tgt.copy(); out[t] = p; t += 1
                    break
                p = p + d / dist * spd * DT + rs.normal(0, 0.15, 3) * [1, 1, 0.3]
                out[t] = p; t += 1
        elif kind == "hover":
            for _ in range(int(s[1])):
                if t >= T_MAX: break
                out[t] = p + rs.normal(0, 0.3, 3); t += 1
        elif kind == "orbit":
            c, rad, secs, spd = np.array(s[1], float), float(s[2]), int(s[3]), float(s[4])
            ang = math.atan2(p[0] - c[0], p[1] - c[1]); w = spd / max(rad, 5.0)
            for _ in range(secs):
                if t >= T_MAX: break
                ang += w * DT
                p = np.array([c[0] + rad * math.sin(ang), c[1] + rad * math.cos(ang), p[2]])
                out[t] = p; t += 1
        elif kind == "wander":
            secs, spd, c, rad = int(s[1]), float(s[2]), np.array(s[3], float), float(s[4])
            hd = rs.uniform(0, 2 * math.pi)
            for _ in range(secs):
                if t >= T_MAX: break
                hd += rs.normal(0, 0.5)
                step = np.array([math.sin(hd), math.cos(hd)]) * spd * DT
                if np.linalg.norm(p[:2] + step - c) > rad:
                    hd = math.atan2(c[0] - p[0], c[1] - p[1])
                    step = np.array([math.sin(hd), math.cos(hd)]) * spd * DT
                p = np.array([p[0] + step[0], p[1] + step[1], max(1.0, p[2] + rs.normal(0, 0.3))])
                out[t] = p; t += 1
    # demo only: when the plan runs out before the window does, a hovering/wandering drone stays put and a drone
    # that is leaving keeps flying out, instead of vanishing mid-air. An approach into the stadium still ends there.
    if DEMO_MODE and segs and 1 < t < T_MAX:
        last = segs[-1]
        if last[0] == "goto" and np.hypot(last[1][0], last[1][1]) > 300:
            v = out[t - 1] - out[t - 2]
            while t < T_MAX and np.hypot(p[0], p[1]) < 8000:
                p = p + v; out[t] = p; t += 1
        elif last[0] in ("hover", "wander", "orbit"):
            while t < T_MAX:
                out[t] = p; t += 1
    return out, seg_t


def tangent_line(start_xy, close, side, r0):
    """Waypoints of a straight pass whose closest distance to midfield is `close`."""
    b = bearing_of(start_xy)
    theta = math.degrees(math.acos(min(1.0, close / max(r0, close + 1e-6))))
    pc = pol(b + side * theta, close)
    d = pc - start_xy
    n = np.linalg.norm(d) + 1e-9
    return pc, pc + d / n * max(r0, 600.0)


def build_case(row, rs):
    """Return list of objects: dict(truth, nominal, kind, is_background, ...)."""
    n_true = int(row["n_objects"]); n = min(n_true, MAX_OBJ)
    b, r0, alt, spd = row["start_bearing_deg"], row["start_range_m"], row["start_alt_m"], row["speed_mps"]
    close, path, target = row["closest_approach_m"], row["path_type"], row["target"]
    start = pol(b, r0)
    side = rs.choice([-1, 1])
    if target == "bowl":
        tgt = pol(rs.uniform(0, 360), rs.uniform(20, 95))
    elif target in ("tailgate_lot", "exit_plaza"):
        tgt = pol(rs.uniform(0, 360), close)
    else:
        tgt = pol(rs.uniform(0, 360), max(close, 0.0))
    listed = row["on_approved_list"] in (True, "True", "wrong_time")
    objs = []
    show_c = None

    def add(st, segs, nominal_segs=None, kind=None, bg=False, label=None, event_seg=None, before=None, lag=5):
        tr, seg_t = run_plan(st, segs, rs)
        nom = run_plan(st, nominal_segs, rs)[0] if nominal_segs is not None else tr
        ev = None
        if event_seg is not None:   # if the change never happens inside the simulated window, the "before" label holds throughout
            ev = (seg_t[event_seg] + lag) if event_seg < len(seg_t) and seg_t[event_seg] < T_MAX else T_MAX + 1
        objs.append(dict(truth=tr, nominal=nom, kind=kind or row["object_type"], bg=bg, label=label, event_t=ev, before=before,
                         plan=[(sg[0], sg[1] if sg[0] == "goto" else None) for sg in segs], seg_t=seg_t))

    def off(i, spacing=6.0):
        k = int(math.ceil(math.sqrt(max(n, 1))))
        return np.array([(i % k - k / 2) * spacing, (i // k - k / 2) * spacing])

    S3 = lambda xy, z: (xy[0], xy[1], z)
    if path in ("transit", "transit_detour"):
        pc, end = tangent_line(start, close, side, r0)
        nomsegs = [("goto", S3(pc, alt), spd), ("goto", S3(end, alt), spd)]
        segs = nomsegs
        if path == "transit_detour":
            mid = (start + pc) / 2; perp = np.array([-(pc - start)[1], (pc - start)[0]]); perp /= np.linalg.norm(perp) + 1e-9
            segs = [("goto", S3(mid + perp * row["route_deviation_m"], alt), spd)] + nomsegs
        add(S3(start, alt), segs, nomsegs)
    elif path == "transit_then_deviate":
        pc, end = tangent_line(start, min(5600.0, r0 * 0.98), side, r0)
        nomsegs = [("goto", S3(pc, alt), spd), ("goto", S3(end, alt), spd)]
        mid = (start + pc) / 2
        inward = -mid / (np.linalg.norm(mid) + 1e-9)
        dev_pt = mid + inward * (np.linalg.norm(mid) - close)
        add(S3(start, alt), [("goto", S3(mid, alt), spd), ("goto", S3(dev_pt, alt), spd), ("hover", 40)], nomsegs, event_seg=1, before="authorized")
    elif path == "box_patrol":
        add(S3(pol(b, min(r0, 60)), alt), [("wander", T_MAX, max(spd, 1.0), (0, 0), 60)])
    elif path == "orbit":
        rad = max(close, 80.0)
        for i in range(n):
            st = pol(b + i * 180, r0)
            add(S3(st, alt), [("goto", S3(pol(b + i * 180, rad), alt), max(spd, 4.0)), ("orbit", (0, 0), rad, T_MAX, max(spd, 3.0))])
    elif path == "formation_show":
        c = pol(rs.uniform(0, 360), min(close, 60.0))
        for i in range(n):
            slot = c + off(i, 5.0)
            st = start + off(i, 5.0) if r0 > 200 else slot + rs.normal(0, 3, 2)
            add(S3(st, alt), [("goto", S3(slot, alt + (i % 4) * 4), max(spd, 2.0)), ("wander", T_MAX, 0.6, slot, 6)])
    elif path == "survey_grid":
        segs = []
        for k in range(8):
            segs.append(("goto", S3(start + np.array([(-1) ** k * 120, k * 25]), alt), max(spd, 2.0)))
        add(S3(start, alt), segs)
    elif path == "flyover":
        end = -start
        for i in range(n):
            o = np.array([i * 25.0, -i * 25.0])
            add(S3(start + o, alt), [("goto", S3(end + o, alt), spd)])
    elif path == "wander_flock":
        pc, end = tangent_line(start, close, side, r0)
        for i in range(n):
            o = rs.normal(0, 18, 2)
            add(S3(start + o, alt + rs.normal(0, 6)), [("goto", S3(pc + o + rs.normal(0, 25, 2), alt + rs.normal(0, 10)), spd * rs.uniform(0.8, 1.2)),
                                                          ("goto", S3(end + o + rs.normal(0, 40, 2), alt), spd * rs.uniform(0.8, 1.2))])
    elif path == "soar_circle":
        add(S3(start, alt), [("orbit", start + rs.normal(0, 30, 2), rs.uniform(30, 80), T_MAX, spd)])
    elif path == "erratic":
        for i in range(n):
            st = start + rs.normal(0, 25, 2)
            add(S3(st, alt + rs.normal(0, 4)), [("wander", T_MAX, max(spd, 1.0), st, 110)])
    elif path == "drift_up":
        wd = pol(rs.uniform(0, 360), 400)
        for i in range(n):
            st = start + rs.normal(0, 12, 2)
            add(S3(st, alt), [("goto", S3(st + wd, alt + 320), max(spd, 1.0))])
    elif path == "ballistic":
        for i in range(n):
            st = start + rs.normal(0, 30, 2)
            add(S3(st, 2.0), [("wait", rs.integers(0, 60)), ("goto", S3(st + rs.normal(0, 15, 2), alt), spd), ("hover", 2)])
    elif path == "drift":
        add(S3(start, alt), [("goto", S3(start + pol(rs.uniform(0, 360), 500), max(alt - 20, 2)), max(spd, 0.5))])
    elif path == "static":
        add(S3(start, max(alt, 1.0)), [("hover", T_MAX)])
    elif path in ("hover", "hover_long"):
        hp = pol(b, max(close, 5.0))
        add(S3(start, alt), [("goto", S3(hp, alt), 6.0), ("hover", T_MAX)])
    elif path == "direct_then_hover":
        add(S3(start, alt), [("goto", S3(tgt, alt), spd), ("hover", T_MAX)])
    elif path == "wander":
        add(S3(start, alt), [("wander", T_MAX, max(spd, 1.0), start, max(60.0, min(250.0, r0 - close)))])
    elif path == "straight_rth":
        pc, end = tangent_line(start, close, side, r0)
        add(S3(start, alt), [("goto", S3(pc, alt), spd), ("goto", S3(end, alt), spd)])
    elif path == "edge_loiter":
        add(S3(start, alt), [("wander", 50, max(spd, 3.0), start, 300), ("hover", 20), ("wander", T_MAX, max(spd, 3.0), start, 300)])
    elif path == "approach_retreat":
        add(S3(start, alt), [("goto", S3(pol(b, close), alt), spd), ("hover", 6), ("goto", S3(start, alt), spd)], event_seg=2, before="threat", lag=8)
    elif path == "approach_loiter_retreat":
        lp = pol(b, close)
        add(S3(start, alt), [("goto", S3(lp, alt), spd), ("wander", int(rs.uniform(12, 40)), 3.0, lp, 60), ("goto", S3(start, alt), spd)], event_seg=2, before="threat", lag=8)
    elif path == "wander_pair":
        c2 = start + pol(rs.uniform(0, 360), rs.uniform(50, 150))
        add(S3(start, alt), [("wander", T_MAX, max(spd, 1.0), start, 180)])
        add(S3(c2, alt + rs.normal(0, 10)), [("wander", T_MAX, max(spd * rs.uniform(0.6, 1.4), 1.0), c2, 180)])
    elif path == "uncontrolled_descent":
        nomsegs = [("wander", T_MAX, 2.0, (0, 0), 60)]
        st = pol(b, min(r0, 60))
        add(S3(st, alt), [("wander", int(rs.uniform(8, 25)), 2.0, (0, 0), 60), ("goto", S3(st + pol(rs.uniform(0, 360), row["route_deviation_m"]), 0.5), rs.uniform(2, 5)), ("hover", T_MAX)], nomsegs, event_seg=1, before="authorized")
    elif path in ("direct", "direct_low"):
        for i in range(n):
            st = start + rs.normal(0, 40, 2) * (i > 0)
            if listed:   # approved fast pass: continues out the other side
                add(S3(st, alt), [("goto", S3(tgt, alt), spd), ("goto", S3(-start, alt), spd)])
            else:
                add(S3(st, alt), [("wait", i * rs.integers(0, 8)), ("goto", S3(tgt + rs.normal(0, 15, 2) * (i > 0), max(alt * 0.4, 8)), spd)])
    elif path == "high_then_dive":
        over = tgt if not listed else pol(rs.uniform(0, 360), max(close, 30.0))
        add(S3(start, alt), [("goto", S3(over + pol(b, 60), alt), spd), ("goto", S3(over, 10.0 if not listed else 40.0), spd * 1.3)] + ([("hover", T_MAX)] if listed else []))
    elif path in ("direct_drop_egress", "direct_release_pass"):
        add(S3(start, alt), [("goto", S3(tgt, alt), spd), ("hover", 4 if path == "direct_drop_egress" else 0), ("goto", S3(pol(b + 180, 1500), alt), spd * 1.2)])
    elif path == "pop_up_direct":
        mid = (start + tgt) / 2
        add(S3(start, 1.0), [("goto", S3(mid, rs.uniform(30, 60)), spd), ("goto", S3(tgt, 12.0), spd)])
    elif path == "box_then_break_out":
        if row.get("concurrent_event") == "approved_light_show":
            c = pol(rs.uniform(0, 360), 30)
            for i in range(n):
                slot = c + off(i, 5.0)
                nomsegs = [("wander", T_MAX, 0.6, slot, 6)]
                add(S3(slot, alt), [("wander", int(rs.uniform(10, 35)), 0.6, slot, 6), ("goto", S3(tgt + rs.normal(0, 10, 2), 15.0), spd)], nomsegs, event_seg=1, before="authorized")
        else:
            nomsegs = [("wander", T_MAX, 3.0, (0, 0), 60)]
            add(S3(pol(b, min(r0, 60)), alt), [("wander", int(rs.uniform(15, 40)), 3.0, (0, 0), 60), ("goto", S3(tgt, 15.0), spd)], nomsegs, event_seg=1, before="authorized")
    elif path == "decoy_plus_direct":
        add(S3(start, alt), [("goto", S3(pol(b, rs.uniform(300, 500)), alt), spd), ("wander", T_MAX, 5.0, pol(b, 400), 120)])
        for i in range(1, n):
            st = pol(b + 180 + rs.uniform(-40, 40), r0)
            add(S3(st, alt), [("wait", rs.integers(8, 30)), ("goto", S3(tgt + rs.normal(0, 15, 2), 12.0), spd * 1.2)])
    elif path == "spotter_plus_direct":
        sp = pol(b, rs.uniform(400, 900))
        add(S3(start, alt), [("goto", S3(sp, alt), spd), ("hover", T_MAX)])
        add(S3(pol(b + rs.uniform(90, 270), r0), alt), [("wait", rs.integers(15, 40)), ("goto", S3(tgt, 12.0), spd * 1.2)])
    elif path == "converge_multi":
        for i in range(n):
            st = pol(b + i * 360.0 / n + rs.uniform(-10, 10), r0 * rs.uniform(0.9, 1.1))
            add(S3(st, alt + rs.normal(0, 8)), [("goto", S3(tgt + rs.normal(0, 25, 2), 12.0), spd * rs.uniform(0.92, 1.08))])
    elif path == "formation_then_split":
        hold = pol(b, 400.0 if not listed else min(r0, 150.0))
        for i in range(n):
            o = off(i, 8.0)
            final = pol(rs.uniform(0, 360), rs.uniform(20, 95)) if not listed else hold + o * 4
            add(S3(start + o, alt), [("goto", S3(hold + o, alt), spd), ("goto", S3(final, 12.0 if not listed else alt), spd)] + ([("hover", T_MAX)] if listed else []))
    elif path in ("formation_hold", "formation_hold_attack"):
        # drones fly in formation to a hold point outside the stadium and keep station there (watching);
        # the attack variant then converges on the bowl together
        hold = pol(b, close)
        for i in range(n):
            o = off(i, 12.0)
            segs = [("goto", S3(hold + o, alt + (i % 3) * 3), spd), ("hover", int(rs.uniform(15, 40)) if path == "formation_hold_attack" else T_MAX)]
            if path == "formation_hold_attack":
                segs.append(("goto", S3(tgt + rs.normal(0, 25, 2), 12.0), spd * 1.2))
                add(S3(start + o, alt), segs, event_seg=2, before="borderline", lag=3)
            else:
                add(S3(start + o, alt), segs)
    elif path == "tailgate_cluster":
        # several unrelated hobbyists flying near the same lot: close together but not coordinated
        c = pol(b, r0)
        for i in range(n):
            st = c + rs.normal(0, 40, 2)
            add(S3(st, rs.uniform(15, 60)), [("wander", T_MAX, rs.uniform(2, 8), c, 150)])
    elif path == "join_formation":
        show_c = pol(rs.uniform(0, 360), 30)
        for i in range(n):
            slot = show_c + off(i + 3, 5.0) + rs.normal(0, 4, 2)
            add(S3(start + rs.normal(0, 30, 2) * (i > 0), alt), [("goto", S3(slot, alt), spd), ("wander", T_MAX, 0.6, slot, 6)])
    elif path == "waves":
        per = int(rs.integers(2, 7)); gap = int(rs.integers(15, 30))
        c = pol(rs.uniform(0, 360), min(close, 60.0))
        for i in range(n):
            w = i // per
            st = start + rs.normal(0, 30, 2)
            if listed:
                slot = c + off(i, 5.0)
                add(S3(st, alt), [("wait", w * gap), ("goto", S3(slot, alt), spd), ("wander", T_MAX, 0.6, slot, 6)])
            else:
                add(S3(st, alt), [("wait", w * gap), ("goto", S3(tgt + rs.normal(0, 25, 2), 12.0), spd)])
    else:
        add(S3(start, alt), [("goto", S3(tgt, alt), spd)])

    # background objects for concurrent events (count as neighbours, not training rows)
    ce = str(row.get("concurrent_event", "none"))
    if ce == "approved_light_show" and path != "box_then_break_out":
        c = show_c if show_c is not None else pol(rs.uniform(0, 360), 30)
        for i in range(10):
            slot = c + off(i, 5.0)
            add(S3(slot, alt), [("wander", T_MAX, 0.6, slot, 6)], kind="multirotor_drone", bg=True, label="authorized")
    elif ce == "flyover":
        fs = pol(rs.uniform(0, 360), 9000)
        for i in range(3):
            add(S3(fs + np.array([i * 25.0, -i * 25.0]), 400), [("wait", rs.integers(5, 40)), ("goto", S3(-fs, 400), 200)], kind="military_jet", bg=True, label="authorized")
    elif ce.endswith("other_hobbyists"):
        for i in range(10):
            st = pol(rs.uniform(0, 360), rs.uniform(300, 2500))
            add(S3(st, rs.uniform(20, 110)), [("wander", T_MAX, rs.uniform(2, 10), st, 200)], kind="multirotor_drone", bg=True, label="unauthorized_careless")
    elif ce == "several_arrivals":
        # 4-8 unrelated drones show up at about the same time from different directions, heading for spots near the stadium
        for i in range(int(rs.integers(4, 9))):
            st = pol(rs.uniform(0, 360), rs.uniform(1200, 2500)); dst = pol(rs.uniform(0, 360), rs.uniform(300, 700))
            add(S3(st, rs.uniform(30, 110)), [("wait", int(rs.integers(0, 15))), ("goto", S3(dst, rs.uniform(25, 80)), rs.uniform(6, 15)), ("wander", T_MAX, rs.uniform(1, 5), dst, 120)],
                kind="multirotor_drone", bg=True, label="unauthorized_careless")
    elif ce == "crossing_track":
        v = objs[0]["truth"]; ok = np.isfinite(v[:, 0])
        if ok.sum() > 20:
            mid = v[ok][len(v[ok]) // 3]
            d0 = pol(rs.uniform(0, 360), 600)
            add((mid[0] + d0[0], mid[1] + d0[1], mid[2]), [("goto", (mid[0] - d0[0], mid[1] - d0[1], mid[2]), 600 * 2 / max(len(v[ok]) // 3 * 2, 10))], kind="multirotor_drone", bg=True, label="unauthorized_careless")
    # home / take-off point of each drone (what Remote ID would report as the operator location)
    for o in objs:
        tr = o["truth"]; ok = np.where(np.isfinite(tr[:, 0]))[0]
        if not len(ok) or o["kind"] not in ("multirotor_drone", "fixed_wing_drone"): o["home"] = None; continue
        p0 = tr[ok[0], :2]
        if path == "straight_rth" and not o["bg"]:
            o["home"] = tr[ok[-1], :2] + rs.normal(0, 30, 2)          # a lost-link drone flies straight back to its home point
        elif tr[ok[0], 2] < 15 or np.hypot(*p0) < 150:
            o["home"] = p0.copy()                                      # launched here (lot, inside the stadium, box over the field)
        else:
            k = ok[min(5, len(ok) - 1)]; hd = tr[k, :2] - p0; nh = np.linalg.norm(hd)
            back = -hd / nh if nh > 1 else pol(rs.uniform(0, 360), 1.0)
            o["home"] = p0 + back * rs.uniform(200, 2500)             # launched somewhere behind where we first see it

    return objs, n_true


# ---- sensors and fusion ---------------------------------------------------------------
def stand_height_old(x, y):
    """Height of the stadium structure at (x, y): open field in the middle, stands rising to the outer wall."""
    ax, ay = np.abs(x), np.abs(y)
    inside = (ax < BOWL_X) & (ay < BOWL_Y)
    field = (ax < 40) & (ay < 68)
    ux = np.clip((ax - 40) / (BOWL_X - 40), 0, 1); uy = np.clip((ay - 68) / (BOWL_Y - 68), 0, 1)
    ew = ux >= uy
    top = np.where(ew, STAND_H["ew"], STAND_H["ns"])
    h = 2.5 + np.maximum(ux, uy) * (top - 2.5)
    return np.where(inside & ~field, h, 0.0)


def stand_height(x, y):
    return geo.height_at(np.asarray(x, float), np.asarray(y, float))


def line_of_sight(sensor_xy, sensor_z, xy, z):
    """True where the straight line from a sensor to the object clears the stadium and the surrounding buildings."""
    xy = np.asarray(xy, float)
    return geo.los((sensor_xy[0], sensor_xy[1], sensor_z), xy[:, 0], xy[:, 1], z)


def line_of_sight_old(sensor_xy, sensor_z, xy, z):
    ok = np.ones(len(xy), bool)
    for f in np.linspace(0.06, 0.96, 16):          # skip the first few metres (the sensor sits on the tower)
        px = sensor_xy[0] + f * (xy[:, 0] - sensor_xy[0]); py = sensor_xy[1] + f * (xy[:, 1] - sensor_xy[1])
        pz = sensor_z + f * (np.nan_to_num(z) - sensor_z)
        ok &= ~(pz < stand_height(px, py))
    return ok


def working_radars(row):
    degs = conditions(row)
    return [] if "no_radar" in degs else [r for r in RADARS if "radar_" + r not in degs and "link_" + r not in degs]


def conditions(row):
    """Sensor failures and challenge conditions for this case, as a set ('fog+no_rf' -> {'fog', 'no_rf'})."""
    return set(str(row["degraded_sensor"]).split("+"))
DELIVERY_SIDS = {1, 2, 26, 27, 71, 74}


def attachment_truth(row, o, kind, rs):
    """What is really hanging under the object: 1 nothing, 2 camera gimbal, 3 carried object (package, canister, sprayer, banner).
    None for things that are not drones (the camera classifies those separately)."""
    if kind not in ("multirotor_drone", "fixed_wing_drone"): return None
    sid = int(row["scenario_id"])
    if not o["bg"]:
        if sid == 36: return 2
        if bool(row["payload_visible"]) or sid in DELIVERY_SIDS or sid == 25: return 3
    if kind == "fixed_wing_drone": return 1 if rs.random() < 0.7 else 2
    return 2 if rs.random() < 0.75 else 1      # most multirotors carry a camera
SIZE_MIX = {
    "delivery": {"medium": 0.6, "large": 0.4}, "broadcast": {"small": 0.1, "medium": 0.7, "large": 0.2},
    "police": {"small": 0.3, "medium": 0.6, "large": 0.1}, "show": {"micro": 0.5, "small": 0.5},
    "research": {"small": 0.4, "medium": 0.5, "large": 0.1}, "hobby": {"micro": 0.45, "small": 0.45, "medium": 0.1},
    "toy": {"micro": 1.0}, "fpv": {"micro": 0.3, "small": 0.7}, "payload": {"medium": 0.6, "large": 0.4},
    "fixed_wing": {"medium": 0.6, "large": 0.4}, "spoof": {"small": 0.6, "medium": 0.4},
    "swarm": {"micro": 0.3, "small": 0.6, "medium": 0.1}, "attack": {"micro": 0.2, "small": 0.55, "medium": 0.25},
    "approved_bg": {"small": 0.4, "medium": 0.5, "large": 0.1}}
SID_MIX = {**{k: "delivery" for k in (1, 2, 26, 27, 71, 74)}, **{k: "broadcast" for k in (3, 47, 69, 72, 73)},
           **{k: "police" for k in (4, 70, 78, 80)}, **{k: "show" for k in (5, 68, 75, 76, 77)},
           **{k: "research" for k in (6, 28, 29, 38)}, 19: "toy", 23: "fpv", 25: "broadcast",
           **{k: "payload" for k in (42, 43, 58)}, 57: "fixed_wing", 46: "spoof", 48: "spoof",
           **{k: "swarm" for k in (49, 51, 52, 53, 54, 55, 56, 81, 82)}}


def size_class(row, o, kind, rs):
    """True size class of a drone (never shown to the model). None for things that are not drones."""
    if kind not in ("multirotor_drone", "fixed_wing_drone"): return None
    if kind == "fixed_wing_drone": mix = "fixed_wing"
    elif o["bg"]: mix = "approved_bg" if o["label"] == "authorized" else "hobby"
    else:
        sid = int(row["scenario_id"]); lab = row["expected_label"]
        mix = SID_MIX.get(sid) or ("attack" if lab == "threat" else "approved_bg" if lab == "authorized" else "hobby")
    m = SIZE_MIX[mix]; ks = list(m)
    return ks[int(rs.choice(len(ks), p=[m[k] for k in ks]))]


def simulate_sensors(row, objs, rs):
    """Pass 1: radar, RF, Remote ID for every object. Pass 2: the four PTZ cameras are shared, so one
    scheduler assigns them second by second. Pass 3: camera identification and position fusion."""
    crowd = row["phase"] in ("first_half", "second_half", "halftime")
    night = row["lighting"] == "night"
    rain = row["weather"] == "rain"
    degs = conditions(row); deg = str(row["degraded_sensor"])
    fog = "fog" in degs; rfi = "rf_interference" in degs
    rf_pd = (RF_PD_CROWD if crowd else RF_PD_QUIET) * (0.35 if rfi else 1.0)
    rid_pd = (RID_PD_CROWD if crowd else RID_PD_QUIET) * (0.35 if rfi else 1.0)
    wx = (0.5 if rain or "cameras_all_rain" in degs else 1.0) * (0.35 if fog else 1.0)
    cam_base = {k: CAM_GRADE_RANGE[CAM_GRADE[k]][1 if night else 0] * wx for k in PODS}
    cam_range = (CAM_RANGE_NIGHT if night else CAM_RANGE_DAY) * wx       # reference (corner camera, small drone)
    radars = [] if "no_radar" in degs else [r for r in RADARS if "radar_" + r not in degs and "link_" + r not in degs]
    rf_nodes = [] if "no_rf" in degs else [k for k in PODS if not ({"rf_" + k, "link_" + k, "rf_all"} & degs)]
    rf_noise_boost = 2.0 if any(d.startswith("rf_") and d != "rf_interference" for d in degs) else 1.0
    rf_pos_boost = 2.5 if rfi else 1.0
    cams = [] if "no_cam" in degs else [k for k in PODS if not ({"camera_" + k, "link_" + k} & degs)]
    if "cameras_out_2" in degs and len(cams) > 2:
        cams = sorted(rs.choice(cams, size=len(cams) - 2, replace=False).tolist())
    rf_on = row["rf_link"] in (True, "True")
    rid = str(row["remote_id"])
    T = len(objs[0]["truth"]) if objs else T_MAX
    tt = np.arange(T)
    # ---------- pass 1: radar, RF, Remote ID ----------
    for o in objs:
        tr = o["truth"]; kind = o["kind"]
        present = np.isfinite(tr[:, 0])
        xy = tr[:, :2]; z = tr[:, 2]
        dist0 = np.hypot(xy[:, 0], xy[:, 1])
        in_bowl = (np.abs(xy[:, 0]) < BOWL_X) & (np.abs(xy[:, 1]) < BOWL_Y)
        o_rf = rf_on if not o["bg"] else kind == "multirotor_drone"
        o_rid = rid if not o["bg"] else ("valid" if o["label"] == "authorized" else "none")
        if kind in ("bird", "bird_or_bat", "balloon", "pyrotechnic", "debris", "camera_only_ghost", "military_jet", "crewed_aircraft"):
            o_rf = False; o_rid = "adsb" if (kind == "crewed_aircraft" and rs.random() < ADSB_SHARE) else "none"
        # control-link activity over time. A lost-link drone transmitted until the link dropped (sometimes before we could
        # hear it). Some radio-silent attackers are flown manually at first, then switched to a pre-programmed route.
        rf_act = np.full(T, bool(o_rf))
        if not o["bg"] and not o_rf and kind in ("multirotor_drone", "fixed_wing_drone"):
            if int(row["scenario_id"]) == 24:
                cut = -1 if rs.random() < 0.5 else int(rs.uniform(5, 40)); rf_act = tt < cut
            elif row["expected_label"] == "threat" and rs.random() < 0.25:
                rf_act = tt < int(rs.uniform(5, 40))
        # radar
        o["size"] = size_class(row, o, kind, rs)
        R_obj = SIZE_CLASSES[o["size"]]["radar_m"] if o["size"] else NONDRONE_RADAR_M.get(kind, 2500.0)
        o["cam_k"] = SIZE_CLASSES[o["size"]]["cam"] if o["size"] else NONDRONE_CAM.get(kind, 1.0)
        rad_det = np.zeros(T, bool); rr_min = np.full(T, np.inf); rad_by = {}; rad_rng = np.full(T, np.inf)
        for r in radars:
            rp = np.array(PODS[r]); rr = np.hypot(xy[:, 0] - rp[0], xy[:, 1] - rp[1])
            pd = np.full(T, RADAR_PD.get(kind, 0.9))
            pd = pd * np.clip((R_obj - np.nan_to_num(rr, nan=1e9)) / (0.25 * R_obj), 0, 1)   # full rate to 75% of the size-class range, zero at it
            pd = np.where((z < 15) & (rr > 300), pd * 0.35, pd)          # ground clutter
            pd = np.where("occluded_by_stands" in degs, pd * 0.4, pd)
            pd = np.where(in_bowl, pd * 0.6, pd)                           # above the bowl: clutter from the crowd and structure below
            los = present & (rr < RADAR_RANGE)
            if los.any(): los[los] = line_of_sight(rp, SITE_Z[r]["radar"], xy[los], z[los])   # real stadium + campus buildings
            d = los & (rs.random(T) < pd)
            rad_by[r] = d; rad_det |= d
            rr_seen = np.where(d, np.nan_to_num(rr, nan=np.inf), np.inf)
            rad_rng = np.minimum(rad_rng, rr_seen); rr_min = np.minimum(rr_min, rr_seen)
        # RF direction finding (only while the link is active)
        n_rf = np.zeros(T, int); M11 = np.zeros(T); M12 = np.zeros(T); M22 = np.zeros(T)
        if rf_act.any():
            for k in rf_nodes:
                kp = np.array(PODS[k]); rr = np.hypot(xy[:, 0] - kp[0], xy[:, 1] - kp[1])
                cand = present & rf_act & (rr < RF_RANGE)
                vis = np.zeros(T, bool)
                if cand.any(): vis[cand] = line_of_sight(kp, SITE_Z[k]["rf"], xy[cand], z[cand])
                p_k = np.where(vis, 1.0, 0.5) * rf_pd / rf_noise_boost        # radio tolerates partial blocking, at a cost
                hk = cand & (rs.random(T) < p_k)
                n_rf += hk.astype(int)
                th = np.arctan2(np.nan_to_num(xy[:, 0] - kp[0]), np.nan_to_num(xy[:, 1] - kp[1])); r2 = np.maximum(np.nan_to_num(rr, nan=1e6), 30.0) ** 2
                w = np.where(hk, 1.0 / r2, 0.0) * np.where(vis, 1.0, 0.25)      # blocked path: multipath, worse bearing
                M11 += w * np.cos(th) ** 2; M22 += w * np.sin(th) ** 2; M12 -= w * np.sin(th) * np.cos(th)
        rf_det = n_rf >= 1; rf_fix = n_rf >= 2
        det_ = M11 * M22 - M12 ** 2
        rf_sd = np.where(rf_fix & (det_ > 0), RF_BEARING_SD * np.sqrt(np.maximum((M11 + M22) / np.where(det_ > 0, det_, 1.0), 0)), np.inf)
        rf_sd = np.maximum(rf_sd, 10.0)                                  # triangulation error (m), from the geometry of the sensors that heard it
        # Remote ID is broadcast by the drone itself, so it continues when the control link is lost
        rid_det = np.zeros(T, bool)
        if o_rid in ("valid", "mismatch_small", "mismatch_brief", "mismatch_large", "cloned") and len(rf_nodes) > 0:
            dr = np.min([np.hypot(xy[:, 0] - PODS[k][0], xy[:, 1] - PODS[k][1]) for k in rf_nodes], axis=0)    # nearest receiver
            fade = np.clip((RID_MAX_M - np.nan_to_num(dr, nan=1e9)) / (RID_MAX_M - RID_FULL_M), 0, 1)
            rid_det = present & (rs.random(T) < rid_pd / rf_noise_boost * fade)
        rid_err = np.zeros(T)
        if o_rid == "mismatch_small": rid_err[:] = rs.uniform(45, 110)
        elif o_rid == "mismatch_large": rid_err[:] = rs.uniform(300, 2500)
        elif o_rid == "mismatch_brief":
            s0 = rs.integers(5, 60); rid_err[s0:s0 + rs.integers(3, 10)] = rs.uniform(80, 400)
        rid_err = rid_err + np.abs(rs.normal(0, 8, T))
        any_det = rad_det | rf_det | rid_det
        if kind == "camera_only_ghost":
            any_det = present & (rs.random(T) < 0.25)
        # micro-Doppler (rotor signature) needs a strong return: it fades with range. Wing beats sometimes look rotor-like.
        md_p = {"multirotor_drone": 0.9, "fixed_wing_drone": 0.5}.get(kind, BIRD_MD if kind in ("bird", "bird_or_bat") else 0.03)
        md_obs = rad_det & (rs.random(T) < md_p * np.clip(1.0 - rr_min / (1.6 * R_obj), 0.1, 1.0))
        o.update(rad_by=rad_by, rad_rng=rad_rng, md_obs=md_obs, present=present, rad_det=rad_det, rf_det=rf_det, rf_fix=rf_fix, rid_det=rid_det, rid_err=rid_err, rid_kind=o_rid,
                 rf_on=o_rf, rf_act=rf_act, any_det=any_det, in_bowl=in_bowl, dist0=dist0, rf_sd=rf_sd)
    # ---------- pass 2: camera scheduler ----------
    pcam = 0.85 * (0.85 if night else 1.0) * (0.6 if rain else 1.0) * (0.5 if "occluded_by_stands" in degs else 1.0) * (0.7 if fog else 1.0)
    nobj = len(objs)
    dcam = {k: np.array([np.sqrt((o["truth"][:, 0] - PODS[k][0]) ** 2 + (o["truth"][:, 1] - PODS[k][1]) ** 2 + (np.nan_to_num(o["truth"][:, 2]) - SITE_Z[k]["cam"]) ** 2)
                         for o in objs]) for k in cams}
    CR = {k: [cam_base[k] * o["cam_k"] for o in objs] for k in cams}    # each camera's range on each object (grade x size)
    for k in cams:                         # a camera can only look at what it has a clear line of sight to
        for i, o in enumerate(objs):
            m = o["present"] & (dcam[k][i] <= CR[k][i])
            if m.any():
                v = np.zeros(len(m), bool); v[m] = line_of_sight(PODS[k], SITE_Z[k]["cam"], o["truth"][m, :2], o["truth"][m, 2])
                dcam[k][i] = np.where(v, dcam[k][i], np.inf)
            else:
                dcam[k][i] = np.inf
    lock = np.zeros((nobj, T), int)          # bitmask of cameras looking at each object (bit per pod in PODS order)
    asg = np.zeros((nobj, T), int)           # bitmask of cameras assigned to it (including while still slewing / focusing)
    sweep_hit = np.zeros((nobj, T), bool)
    tracked = np.zeros(nobj, bool); last_look = np.full(nobj, -999); looked = np.zeros(nobj, bool)
    assign = {k: None for k in cams}; since = {k: 0 for k in cams}
    bit = {k: 1 << i for i, k in enumerate(PODS)}
    sweeping = crowd or str(row["concurrent_event"]) == "approved_light_show" or int(row["scenario_id"]) in (5, 68, 76, 77)
    for t in range(T):
        live = [i for i, o in enumerate(objs) if o["present"][t]]
        for i in live:
            if objs[i]["any_det"][t] or sweep_hit[i, max(t - 1, 0)]: tracked[i] = True
        cand = [i for i in live if tracked[i] and objs[i]["kind"] != "camera_only_ghost"]
        # tasking priority (a fixed rule, not the ML model): urgency first, then anything never identified
        def prio(i):
            o = objs[i]; tr = o["truth"]; d = o["dist0"][t]
            p0 = tr[max(t - 3, 0), :2]; vr = (np.hypot(*p0) - d) / 3.0 if np.isfinite(p0).all() else 0.0   # closing speed
            ttb = (d - 120) / vr if vr > 1 else 999.0
            return (100.0 / (1.0 + max(ttb, 0) / 30.0) + 30 * o["in_bowl"][t] + 25 * (not looked[i])
                    + 15 * (not o["rf_det"][max(t - 10, 0):t + 1].any()) + min(20.0, (t - last_look[i]) / 3.0))
        pr = {i: prio(i) for i in cand}
        # keep a camera on its track until the minimum dwell is over (and while the track is still the best use of it)
        for k in cams:
            a = assign[k]
            if a is not None and (a not in pr or dcam[k][a, t] > CR[k][a]): assign[k] = None
        taken = {assign[k] for k in cams if assign[k] is not None}
        order = sorted(pr, key=lambda i: -pr[i])
        for k in cams:
            a = assign[k]
            if a is not None and t - since[k] >= CAM_DWELL_S:
                better = [i for i in order if i not in taken and dcam[k][i, t] <= CR[k][i] and pr[i] > pr[a] + 10]
                if better: taken.discard(a); assign[k] = None
        taken = {assign[k] for k in cams if assign[k] is not None}
        for i in order:                                   # one camera per track, highest priority first, nearest free camera
            if i in taken: continue
            free = [k for k in cams if assign[k] is None and dcam[k][i, t] <= CR[k][i]]
            if not free: continue
            k = min(free, key=lambda k: dcam[k][i, t] / CR[k][i]); assign[k] = i; since[k] = t; taken.add(i)   # best view, relative to each camera's range
        for k in cams:                                    # spare cameras: second camera on the top track close in (crossed bearings)
            if assign[k] is None:
                close = [i for i in order if dcam[k][i, t] <= min(CR[k][i], 600) and (objs[i]["in_bowl"][t] or objs[i]["dist0"][t] < 400)]
                if close: assign[k] = close[0]; since[k] = t
        for k in cams:
            a = assign[k]
            if a is not None:
                asg[a, t] |= bit[k]
                if t - since[k] >= CAM_ACQ_S: lock[a, t] |= bit[k]
            elif sweeping:                                # idle camera sweeps the airspace over the bowl
                for i in live:
                    if objs[i]["in_bowl"][t] and objs[i]["kind"] not in ("rf_only_ghost",) and np.isfinite(dcam[k][i, t]) and rs.random() < CAM_SWEEP_PD:
                        sweep_hit[i, t] = True
        for i in range(nobj):
            if lock[i, t]: last_look[i] = t; looked[i] = True
    for o in objs: o["cams_ok"] = list(cams); o["cam_range"] = float(cam_range)
    for i, o in enumerate(objs):           # look distance as a share of the looking camera's range (1.0 = edge of its range)
        cdn = np.full(T, np.inf)
        for k in cams:
            on = (lock[i] & bit[k]) > 0; cdn = np.where(on, np.minimum(cdn, dcam[k][i] / CR[k][i]), cdn)
        o["cam_dist_n"] = cdn
    # ---------- pass 3: what the cameras see, and fused position ----------
    for oi, o in enumerate(objs):
        tr = o["truth"]; kind = o["kind"]; xy = tr[:, :2]; present = o["present"]; dist0 = o["dist0"]
        hold = rs.random(T) < pcam
        cam_det = (lock[oi] > 0) & present & hold
        if kind == "rf_only_ghost": cam_det[:] = False          # camera looks where RF points and finds nothing
        if kind == "camera_only_ghost": cam_det = o["any_det"].copy()
        n_cam = np.array([bin(m).count("1") for m in lock[oi]]) * cam_det
        o["cam_dist_any"] = np.min([dcam[k][oi] for k in cams], axis=0) if cams else np.full(T, np.inf)
        cd = np.full(T, np.inf)
        for k in cams:
            on = (lock[oi] & bit[k]) > 0; cd = np.where(on, np.minimum(cd, dcam[k][oi]), cd)
        det_all = o["any_det"] | sweep_hit[oi] | cam_det
        first = int(np.argmax(det_all)) if det_all.any() else -1
        # what the camera sees hanging under the object, each second it looks (image classifier with a confidence)
        att_true = attachment_truth(row, o, kind, rs)
        att_obs = np.zeros(T, int); att_conf = np.zeros(T)       # 0 no look, 1 nothing attached, 2 camera gimbal, 3 carried object, 4 cannot tell
        if att_true is not None:
            q = np.clip(1.0 - np.where(np.isfinite(o["cam_dist_n"]), o["cam_dist_n"], 1.0), 0.0, 1.0)
            p_ok = (0.25 + 0.7 * q) * (0.85 if night else 1.0) * (0.8 if rain else 1.0)
            u = rs.random(T); u2 = rs.random(T)
            wrong = {1: 2, 2: 3, 3: 2}[att_true]                  # the likely confusion: nothing <-> gimbal, gimbal <-> carried object
            obs = np.where(u < p_ok, att_true, np.where(u2 < 0.6, 4, wrong))
            conf = np.where(obs == att_true, 0.5 + 0.45 * q * rs.random(T) + 0.05, np.where(obs == 4, 0.1 + 0.2 * rs.random(T), 0.35 + 0.3 * rs.random(T)))
            att_obs = np.where(cam_det, obs, 0); att_conf = np.where(cam_det, conf, 0.0)
        # fused position: the measured source with the smallest expected error wins; claims only if nothing was measured
        fused = np.full((T, 2), np.nan); src = np.zeros(T, int)   # 1 radar, 2 RF fix, 3 camera, 4 Remote ID claim
        noise = rs.normal(0, 1, (T, 2))
        s_rad = np.where(o["rad_det"], np.sqrt(RADAR_RANGE_ACC ** 2 + (np.where(np.isfinite(o["rad_rng"]), o["rad_rng"], 0) * RADAR_ANG_ACC) ** 2), np.inf)
        s_rf = np.where(o["rf_fix"], o["rf_sd"] * (2.0 if "position_conflict" in degs else 1.0) * rf_pos_boost, np.inf)
        s_cam = np.where(cam_det, np.where(n_cam >= 2, 3.0 + 0.01 * cd, 5.0 + 0.12 * cd), np.inf)   # two cameras cross bearings; one camera has a poor range estimate
        S = np.vstack([s_rad, s_rf, s_cam]); best = S.argmin(axis=0); sbest = S.min(axis=0)
        m = np.isfinite(sbest); fused[m] = xy[m] + noise[m] * sbest[m, None]; src[m] = best[m] + 1
        m = (src == 0) & o["rid_det"]; fused[m] = xy[m] + noise[m] * 12.0 + (o["rid_err"][m] / np.sqrt(2))[:, None]; src[m] = 4
        pos_sd = np.where((src >= 1) & (src <= 3), sbest, np.where(src == 4, 12.0, np.nan))   # the tracker's own estimate of its position error (m)
        if "position_conflict" in degs:
            m = src == 1; fused[m] += rs.normal(0, 35, (m.sum(), 2))
        o.update(pos_sd=pos_sd, cam_asg=asg[oi], cam_det=cam_det, cam_lock=lock[oi], fused=fused, src=src, first=first, att_obs=att_obs, att_conf=att_conf, cam_dist=cd)
        # what the camera says the object is, each second it looks: 1 drone, 2 bird, 3 aircraft or other, 4 cannot tell
        ctrue = 1 if kind in ("multirotor_drone", "fixed_wing_drone") else 2 if kind in ("bird", "bird_or_bat") else 3
        qc = np.clip(1.0 - np.where(np.isfinite(o["cam_dist_n"]), o["cam_dist_n"], 1.0), 0.0, 1.0)
        if kind == "camera_only_ghost": qc = np.full(T, 0.1)        # a blur right at the lens
        pc_ok = (0.5 + 0.45 * qc) * (0.8 if night else 1.0) * (0.85 if rain else 1.0) * (0.7 if fog else 1.0) * CAM_ID_SCALE
        if kind in ("crewed_aircraft", "military_jet"): pc_ok = np.maximum(pc_ok, 0.9)   # big and obvious
        u = rs.random(T); u2 = rs.random(T); confuse = {1: 2, 2: 1, 3: 2}[ctrue]
        cls = np.where(u < pc_ok, ctrue, np.where(u2 < 0.6, 4, confuse))
        cconf = np.where(cls == ctrue, 0.5 + 0.45 * qc * rs.random(T) + 0.05, np.where(cls == 4, 0.1 + 0.2 * rs.random(T), 0.35 + 0.3 * rs.random(T)))
        o["cls_obs"] = np.where(cam_det, cls, 0); o["cls_conf"] = np.where(cam_det, cconf, 0.0)
        fl = np.where(o["cls_obs"] > 0)[0]
        o["cam_class"] = {1: "drone", 2: "bird", 3: "aircraft" if kind in ("crewed_aircraft", "military_jet") else "other", 4: "unknown"}[int(o["cls_obs"][fl[0]])] if len(fl) else "unknown"   # first look (event log)
        o["hij_flag"] = rs.random() < 0.7
        # where Remote ID says the operator / take-off point is (standard Remote ID broadcasts the control-station location)
        o["home_claim"] = (np.asarray(o["home"]) + rs.normal(0, 25, 2)) if o.get("home") is not None else None
    return objs


FEATURES = ["track_age_s", "dist_m", "min_dist_m", "closing_mps", "max_closing_mps", "speed_mps", "speed_std", "alt_m",
            "alt_rate_mps", "heading_to_stadium", "time_to_bowl_s", "frac_hover", "straightness", "radar_frac", "rf_frac",
            "rid_seen", "rid_pos_err_m", "seen_not_heard", "adsb", "on_manifest", "manifest_dev_m", "max_manifest_dev_m", "outside_plan_s", "manifest_time_ok",
            "duplicate_id", "cam_seen", "cam_drone", "cam_bird", "cam_other", "cam_att_none", "cam_att_gimbal", "cam_att_object", "cam_att_conf", "cam_att_looks", "cam_look_dist_m", "micro_doppler",
            "n_near_150m", "n_co_moving", "n_inbound", "n_tracks", "over_bowl", "receding_s", "dist_gain_m", "rf_link_changed", "toward_home", "home_dist_m", "rf_lost_after_heard", "show_scheduled", "is_night",
            "radar_degraded", "rf_degraded", "cam_degraded"]


def features_at(row, objs, oi, t_abs, rs):
    o = objs[oi]; f0 = o["first"]
    sl = slice(f0, t_abs + 1)
    fu = o["fused"][sl]; ok = np.isfinite(fu[:, 0])
    if ok.sum() < 3:
        return None
    idx = np.where(ok)[0]
    pos = fu[ok]; tt = idx.astype(float)
    d = np.hypot(pos[:, 0], pos[:, 1])
    w = min(6, len(pos))
    def vel(a, b):
        dtv = max(tt[b] - tt[a], 1.0); return (pos[b] - pos[a]) / dtv
    v_now = vel(len(pos) - w, len(pos) - 1)
    speed_now = float(np.hypot(*v_now))
    steps = np.hypot(*np.diff(pos, axis=0).T) / np.maximum(np.diff(tt), 1.0) if len(pos) > 1 else np.array([0.0])
    closing = -(d[-1] - d[len(d) - w]) / max(tt[-1] - tt[len(d) - w], 1.0)
    cl_all = -np.diff(d) / np.maximum(np.diff(tt), 1.0) if len(d) > 1 else np.array([0.0])
    k = 5
    cl_s = np.convolve(cl_all, np.ones(k) / k, mode="valid") if len(cl_all) >= k else cl_all
    to_c = -pos[-1] / (d[-1] + 1e-6)
    hts = float(np.dot(v_now, to_c) / (speed_now + 1e-6)) if speed_now > 0.5 else 0.0
    alt_tr = o["truth"][sl][:, 2][ok] + rs.normal(0, 3.0, ok.sum())
    alt_rate = (alt_tr[-1] - alt_tr[len(alt_tr) - w]) / max(tt[-1] - tt[len(alt_tr) - w], 1.0)
    path_len = float(np.hypot(*np.diff(pos, axis=0).T).sum()) if len(pos) > 1 else 0.0
    straight = float(np.hypot(*(pos[-1] - pos[0])) / path_len) if path_len > 5 else 0.0
    edge = max(d[-1] - 110.0, 0.0)
    ttb = edge / closing if closing > 0.5 else 999.0
    n_s = t_abs - f0 + 1
    radar_frac = o["rad_det"][sl].sum() / n_s; rf_frac = o["rf_det"][sl].sum() / n_s
    # ghost check (alert rule only, not a model input): seconds in the last 10 where a working radar had a clear line of
    # sight to where the track is estimated to be, within range, and still reported nothing
    radar_clear_miss = 0; radar_opp = 0
    wr = working_radars(row)
    if wr:
        k0 = max(f0, t_abs - 9); fz = o["fused"][k0:t_abs + 1]; zz = alt_tr[-len(fz):] if len(alt_tr) >= len(fz) else np.full(len(fz), alt_tr[-1])
        good = np.isfinite(fz[:, 0])
        if good.any():
            seen = np.zeros(good.sum(), bool)
            for r in wr:
                rp = PODS[r]; inr = np.hypot(fz[good, 0] - rp[0], fz[good, 1] - rp[1]) < RADAR_GHOST_M
                v = line_of_sight(rp, SITE_Z[r]["radar"], fz[good], np.maximum(zz[good], 20.0)) & inr
                seen |= v
            radar_opp = int(seen.sum())
            radar_clear_miss = 0 if o["rad_det"][max(f0, t_abs - 9):t_abs + 1].any() else radar_opp
    rid_seen = bool(o["rid_det"][sl].any())
    rid_err = float(np.median(o["rid_err"][sl][o["rid_det"][sl]])) if rid_seen else -1.0
    listed = row["on_approved_list"] if not o["bg"] else (o["label"] == "authorized")
    lst_true = listed in (True, "True")
    on_man = 1 if (lst_true or listed == "wrong_time") else 0
    if o["kind"] == "crewed_aircraft": on_man = 1
    man_dev = -1.0; time_ok = 1 if lst_true else 0; dup = 0; max_dev = -1.0; out_s = 0
    if on_man:
        nom = o["nominal"][:, :2]; nm = nom[np.isfinite(nom[:, 0])]
        if len(nm):
            dd = np.sqrt(((pos[:, None, :] - nm[None, :, :]) ** 2).sum(-1)).min(1)      # distance from the filed plan, whole track
            psd = np.nan_to_num(o["pos_sd"][sl][ok], nan=0.0) if "pos_sd" in o else np.zeros(len(dd))
            dd = np.maximum(dd - MANIFEST_SIGMA * psd, 0.0)                              # only what exceeds the position uncertainty counts as deviation
            k5 = np.convolve(dd, np.ones(5) / 5, mode="valid") if len(dd) >= 5 else dd  # smoothed so one noisy fix does not count
            man_dev = float(dd[-1]); max_dev = float(k5.max()); out_s = int((k5 > 40).sum())
    if listed == "cloned_id" or o["rid_kind"] == "cloned":
        if rid_seen:
            on_man = 1; time_ok = 1; dup = 1 if rs.random() < 0.8 else 0
            man_dev = float(np.hypot(pos[-1, 0] - 30, pos[-1, 1] + 20))    # real drone is in its box over the field
            alld = np.hypot(pos[:, 0] - 30, pos[:, 1] + 20); max_dev = float(alld.max()); out_s = int((alld > 40).sum())
    # lost-link clue 1: heading toward the home point that Remote ID reports
    toward_home, home_dist = 0.0, -1.0
    if rid_seen and o.get("home_claim") is not None:
        hv = o["home_claim"] - pos[-1]; home_dist = float(np.hypot(*hv))
        if speed_now > 2.0 and home_dist > 1.0: toward_home = float(np.dot(hv, v_now) / (home_dist * speed_now))
    # lost-link clue 2: we heard the control link earlier in this track, and it has been silent for 8+ s while still tracked
    rfw = o["rf_det"][f0:t_abs + 1]
    rf_lost = int(len(rfw) > 10 and rfw[:-8].sum() >= 2 and not rfw[-8:].any())
    cam_seen = bool(o["cam_det"][sl].any())
    # camera identification: every look so far votes, weighted by confidence; needs at least half the evidence to agree
    co = o["cls_obs"][sl]; cf = o["cls_conf"][sl]
    cv = np.array([cf[co == c].sum() for c in (1, 2, 3)])
    cc = "none"
    if cv.sum() > 0 and cv.max() / (cv.sum() + cf[co == 4].sum()) >= 0.5: cc = {0: "drone", 1: "bird", 2: "other"}[int(cv.argmax())]
    mdw = o["md_obs"][sl]; radw = o["rad_det"][sl]
    rad_idx = np.where(radw)[0][-15:]                       # the last 15 radar returns, for the alert rule's rotor check
    md_hits = int(mdw[rad_idx].sum()); md_returns = len(rad_idx)
    cam_looks = int((co > 0).sum())
    # identification range (DRI): a "drone" call only counts toward confirmation from a look close enough to resolve the object
    cdw = o["cam_dist_n"][sl] if "cam_dist_n" in o else np.full(len(co), np.inf)
    cam_drone_id_looks = int(((co == 1) & (np.nan_to_num(cdw, nan=np.inf) <= CAM_ID_FRAC)).sum())
    # combine every camera look so far: each look votes for what it saw, weighted by its confidence
    ao = o["att_obs"][sl]; ac = o["att_conf"][sl]; looks = (ao >= 1) & (ao <= 3)
    votes = np.array([ac[ao == c].sum() for c in (1, 2, 3)])
    if votes.sum() > 0:
        best_att = int(votes.argmax()) + 1; best_conf = float(votes.max() / votes.sum())      # share of the evidence that agrees
        best_dist = float(o["cam_dist"][sl][looks].min())
    else:
        best_att, best_conf, best_dist = 0, 0.0, -1.0
    n_looks = int(looks.sum())
    # neighbours at this moment
    near = comove = inbound = ntr = 0
    for j, q in enumerate(objs):
        if j == oi or q["first"] < 0 or q["first"] > t_abs: continue
        qf = q["fused"][max(q["first"], t_abs - 6):t_abs + 1]; qo = np.isfinite(qf[:, 0])
        if qo.sum() < 1: continue
        ntr += 1
        qp = qf[qo]; sep = float(np.hypot(*(qp[-1] - pos[-1])))
        if sep < 150: near += 1
        if qo.sum() >= 2:
            qv = (qp[-1] - qp[0]) / max(np.where(qo)[0][-1] - np.where(qo)[0][0], 1)
            if sep < 400 and np.hypot(*(qv - v_now)) < 3.0 and speed_now > 2.0: comove += 1
            qd = np.hypot(*qp[-1]); qc = -(qd - np.hypot(*qp[0])) / max(np.where(qo)[0][-1] - np.where(qo)[0][0], 1)
            if qc > 4.0 and closing > 4.0: inbound += 1
    degs = conditions(row)
    show = 1 if (row["scenario_id"] in (5, 68, 76, 77) or str(row["concurrent_event"]) == "approved_light_show") else 0
    over_bowl = int(abs(pos[-1, 0]) < BOWL_X and abs(pos[-1, 1]) < BOWL_Y)
    rec = 0
    for kk in range(len(cl_all) - 1, -1, -1):
        if cl_all[kk] < -0.5: rec += 1
        else: break
    hij = int(row["scenario_id"] in (47, 80) and o.get("event_t") is not None and t_abs >= o["event_t"] and o.get("hij_flag", False))
    if rs.random() < 0.005: hij = 1
    return dict(track_age_s=n_s, dist_m=d[-1], min_dist_m=d.min(), closing_mps=closing, max_closing_mps=float(cl_s.max()) if len(cl_s) else 0.0,
                speed_mps=speed_now, speed_std=float(np.std(steps)), alt_m=float(alt_tr[-1]), alt_rate_mps=float(alt_rate),
                heading_to_stadium=hts, time_to_bowl_s=min(ttb, 999.0), frac_hover=float((steps < 1.5).mean()), straightness=straight,
                radar_frac=radar_frac, rf_frac=rf_frac, radar_clear_miss=radar_clear_miss, radar_opp=radar_opp, rid_seen=int(rid_seen), rid_pos_err_m=rid_err,
                seen_not_heard=int(radar_frac > 0.5 and rf_frac < 0.1 and not rid_seen), adsb=int(o["rid_kind"] == "adsb"),
                on_manifest=on_man, manifest_dev_m=man_dev, max_manifest_dev_m=max_dev, outside_plan_s=out_s, manifest_time_ok=time_ok, duplicate_id=dup,
                cam_seen=int(cam_seen), cam_drone=int(cc == "drone"), cam_drone_id_looks=cam_drone_id_looks, cam_bird=int(cc == "bird"), cam_other=int(cc in ("other", "aircraft")),
                cam_att_none=int(best_att == 1), cam_att_gimbal=int(best_att == 2), cam_att_object=int(best_att == 3), cam_att_conf=best_conf, cam_att_looks=n_looks, cam_look_dist_m=best_dist, micro_doppler=int(mdw.sum() >= 2 and mdw.sum() >= 0.3 * max(radw.sum(), 1)),
                n_near_150m=near, n_co_moving=comove, n_inbound=inbound, n_tracks=ntr, over_bowl=over_bowl, receding_s=rec, dist_gain_m=float(d[-1] - d.min()), rf_link_changed=hij, md_hits=md_hits, md_returns=md_returns, cam_looks=cam_looks, toward_home=toward_home, home_dist_m=home_dist, rf_lost_after_heard=rf_lost, show_scheduled=show,
                is_night=int(row["lighting"] == "night"),
                radar_degraded=int(any(d.startswith(("radar", "link", "no_radar")) for d in degs)),
                rf_degraded=int(any(d.startswith(("rf", "link", "no_rf")) for d in degs)),
                cam_degraded=int(any(d.startswith(("camera", "link", "no_cam", "fog")) for d in degs)))


def simulate_case(row, keep_tracks=False):
    rs = np.random.default_rng(int(row["seed"]))
    objs, n_true = build_case(row, rs)
    objs = simulate_sensors(row, objs, rs)
    rows = []; stats = dict(case_id=row["case_id"], detected=0, objects=0, first_detect_range_m=np.nan, warning_s=np.nan, starts_inside=0, detect_delay_s=np.nan)
    fg = [i for i, o in enumerate(objs) if not o["bg"]]
    stats["objects"] = len(fg)
    order_sz = ["micro", "small", "medium", "large"]
    szs = [objs[i].get("size") for i in fg if objs[i].get("size")]
    stats["size"] = min(szs, key=order_sz.index) if szs else "none"      # smallest drone in the case (hardest for radar)
    pick = fg if len(fg) <= 4 else list(rs.choice(fg, 4, replace=False))
    for i in fg:
        o = objs[i]
        if o["first"] < 0: continue
        stats["detected"] += 1
        tr = o["truth"]; d = np.hypot(tr[:, 0], tr[:, 1])
        r_first = float(d[o["first"]])
        if not np.isfinite(stats["first_detect_range_m"]) or r_first > stats["first_detect_range_m"]:
            stats["first_detect_range_m"] = r_first
        t0 = int(np.argmax(o["present"]))
        dd = float(o["first"] - t0)
        if not np.isfinite(stats["detect_delay_s"]) or dd > stats["detect_delay_s"]: stats["detect_delay_s"] = dd
        if d[t0] < 150: stats["starts_inside"] = 1
        inside = np.where(np.isfinite(d) & (d < 120))[0]
        if len(inside) and d[t0] >= 150:
            wsec = float(inside[0] - o["first"])
            if not np.isfinite(stats["warning_s"]) or wsec < stats["warning_s"]: stats["warning_s"] = wsec
    for i in pick:
        o = objs[i]
        if o["first"] < 0: continue
        last = np.where(o["present"])[0]
        if not len(last): continue
        cps = list(CHECKPOINTS)
        if o.get("event_t") is not None:     # also sample just after the behaviour changes (hijack, turn-back, failure)
            cps += [int(o["event_t"] - o["first"] + k) for k in (2, 4, 6, 9, 13, 18, 25)]
        for cp in sorted(set(c for c in cps if c >= 5)):
            t_abs = o["first"] + cp
            if t_abs > last[-1] or t_abs >= T_MAX: break
            f = features_at(row, objs, i, t_abs, rs)
            if f is None: continue
            lab = row["expected_label"]
            if o.get("event_t") is not None and o.get("before") and t_abs < o["event_t"]: lab = o["before"]
            f.update(case_id=row["case_id"], scenario_id=int(row["scenario_id"]), obj=i, checkpoint_s=cp,
                     label=lab, case_label=row["expected_label"], is_swarm=int(row["is_swarm"] in (True, "True")), n_objects_true=n_true)
            rows.append(f)
    return rows, stats, (objs if keep_tracks else None)


def run_all(cases_csv="cuas_test_cases.csv", out="features.csv"):
    cases = pd.read_csv(cases_csv)
    allrows, allstats = [], []
    for _, r in cases.iterrows():
        rows, st, _ = simulate_case(r.to_dict())
        allrows += rows; st.update(scenario_id=r["scenario_id"], label=r["expected_label"], group=r["group"]); allstats.append(st)
    df = pd.DataFrame(allrows); df.to_csv(out, index=False)
    sd = pd.DataFrame(allstats); sd.to_csv("case_stats.csv", index=False)
    return df, sd


if __name__ == "__main__":
    import time
    t = time.time(); df, sd = run_all()
    print("cases", len(sd), "feature rows", len(df), "time %.0fs" % (time.time() - t))
    print(df.label.value_counts().to_dict())
    nd = sd[sd.detected == 0]
    print("cases with nothing detected:", len(nd)); print(nd.groupby("label").size().to_dict())
