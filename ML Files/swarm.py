"""Swarm module: finds groups of tracks acting together and decides whether a group is a concern.

Runs after tracking and the per-drone model. Every track keeps its own ID; a suspected group gets a
separate group ID. All rules are fixed and explainable (no learning), so each flag can be justified.

Links between two tracks (checked every second, from the smoothed fused positions):
  co-moving       within 300 m, flying with nearly the same velocity      (similar speed / heading)
  holding         within 200 m, both nearly stationary                     (keeping station together)
  shared target   both closing on the stadium, aimed within 400 m of it,
                  arriving within 45 s of each other                       (shared destination, any direction)
A link counts only if it held for 7 of the last 10 seconds (persistence).
Groups are connected sets of persistent links: 2+ tracks if a shared-target link is involved, else 3+.

Indicators reported per group (the slide's list):
  similar movement (velocity spread), consistent spacing (spacing variation), shared destination
  (aim-point spread, arrival spread), simultaneous zone entry, split/merge events, persistence,
  radio pattern, authorization, drone confirmation, distance / time to spectators.

Group status (what the operator gets):
  approved group           mostly on the manifest, every confirmed drone in it is on the manifest
  intruder in approved grp approved group containing a member that is not on the manifest         -> alert
  coordinated, unconfirmed no member confirmed as a drone (typically a bird flock)                   -> watch
  suspicious formation     confirmed unlisted drones (2+, and 40%+ of the group) coordinating, not heading for people -> alert
  hostile swarm            the same, converging on the stadium                                        -> alarm if urgent, else alert
"""
import math
from collections import deque

CO_D, CO_DV = 300.0, 2.5          # co-moving: max distance (m), max velocity difference (m/s, or 25% of speed)
HOLD_D, HOLD_V = 200.0, 2.5       # holding: max distance (m), max speed (m/s)
AIM_R, AIM_D, ETA_DT = 550.0, 400.0, 45.0   # shared target: aimed within AIM_R of midfield, aim points within AIM_D, arrivals within ETA_DT s
SPACING_SD, SPACING_FRAC = 15.0, 0.15   # steady spacing: distance varies by less than 15 m or 15% over the window
AIM_DRIFT = 150.0                 # steady path: the predicted aim point moves less than this over the window
PERSIST, WINDOW = 7, 10           # link must hold PERSIST of the last WINDOW seconds
URGENT_S, URGENT_M = 180.0, 1000.0  # hostile swarm becomes an ALARM within this time / distance (swarms get a longer horizon than one drone)
ENTRY_R, ENTRY_DT = 1500.0, 20.0  # simultaneous entry: two or more members cross ENTRY_R within ENTRY_DT s
RECENT_S = 15                     # group decisions use drone confirmation from the last 15 s, not a confirmation latched long ago
BIRD_SHARE = 0.3                  # if the camera identifies 30%+ of a group as birds, treat it as birds
CONF_SHARE = 0.5                  # a hostile/suspicious group needs at least this share of members confirmed as drones 
HOLD_DOWN = 10                    # a group's status steps down only after the lower status has held this long (s)
GAP_S = 8                         # a group survives a gap in its links of up to this many seconds
MIN_AGE = 5                       # seconds a track must exist before it can join a group

STATUS_LEVEL = {"approved group": 0, "coordinated, unconfirmed": 1, "coordinated group": 1,
                "suspicious formation": 2, "intruder in approved group": 2, "hostile swarm (forming)": 2, "hostile swarm": 3}


def smooth(tr, T):
    """Alpha-beta filter over the fused fixes (same as the demo page): position and velocity per second."""
    sx, sy, vx_, vy_ = [None] * T, [None] * T, [0.0] * T, [0.0] * T
    x = None; y = vx = vy = 0.0; miss = 0; cnt = 0; px = py = 0.0; pk = 0
    for k in range(tr["first"], T):
        if tr["tx"][k] is None: x = None; continue
        zx, zy = tr["fx"][k], tr["fy"][k]
        if x is None:
            if zx is None: continue
            x, y, vx, vy, miss, cnt, px, py, pk = zx, zy, 0.0, 0.0, 0, 1, zx, zy, k
        else:
            x += vx; y += vy
            if zx is not None:
                cnt += 1
                if cnt == 2: vx = (zx - px) / (k - pk); vy = (zy - py) / (k - pk)
                rx, ry = zx - x, zy - y; x += 0.3 * rx; y += 0.3 * ry; vx += 0.06 * rx; vy += 0.06 * ry; miss = 0
            else:
                miss += 1
                if miss > 8: x = None; continue
        sx[k], sy[k], vx_[k], vy_[k] = x, y, vx, vy
    return sx, sy, vx_, vy_


def aim(x, y, vx, vy):
    """Where a straight-line path passes closest to midfield: (distance of that point from midfield, seconds to get there, point)."""
    s2 = vx * vx + vy * vy
    if s2 < 1.0: return None
    tca = -(x * vx + y * vy) / s2
    if tca <= 0: return None                       # moving away
    ax, ay = x + vx * tca, y + vy * tca
    return math.hypot(ax, ay), tca, (ax, ay)


def link_type(a, b, steady_spacing, aim_steady_a, aim_steady_b):
    (xa, ya, vxa, vya), (xb, yb, vxb, vyb) = a, b
    d = math.hypot(xa - xb, ya - yb); sa, sb = math.hypot(vxa, vya), math.hypot(vxb, vyb)
    # formations (holding station or flying together) must also keep steady spacing; drones that merely happen to be near each other do not
    if d < HOLD_D and sa < HOLD_V and sb < HOLD_V and steady_spacing: return "holding"
    if d < CO_D and max(sa, sb) > HOLD_V and math.hypot(vxa - vxb, vya - vyb) < max(CO_DV, 0.25 * max(sa, sb)) and steady_spacing: return "co-moving"
    # shared target: both on steady paths (aim point not wandering) that end at the same place at about the same time
    if not (aim_steady_a and aim_steady_b): return None
    ia, ib = aim(*a), aim(*b)
    if ia and ib and ia[0] < AIM_R and ib[0] < AIM_R and abs(ia[1] - ib[1]) < ETA_DT and math.hypot(ia[2][0] - ib[2][0], ia[2][1] - ib[2][1]) < AIM_D \
            and (math.hypot(xa, ya) - math.hypot(xa + vxa, ya + vya)) > 3 and (math.hypot(xb, yb) - math.hypot(xb + vxb, yb + vyb)) > 3:
        return "shared target"
    return None


def analyze(tracks, T):
    """Returns (groups_by_second, swarm_level, group_of): groups_by_second[t] = list of group dicts;
    swarm_level[i][t] = 0-3 level the group status gives track i; group_of[i][t] = group id or None."""
    n = len(tracks)
    st = [smooth(tr, T) for tr in tracks]
    hist = {}                                       # (i, j) -> deque of link types (None = no link)
    dhist = {}                                      # (i, j) -> recent distances, for steady spacing
    ahist = {i: deque(maxlen=WINDOW) for i in range(n)}   # recent aim points per track, for steady paths
    chist = {}                                      # group id -> recent centroid distances, for "group closing on the stadium"
    prev_groups, next_id, first_seen, entered = [], 1, {}, {}
    status_hold = {}                                # gid -> (status, time a lower status started or None)
    alive = {}                                      # gid -> (last group dict, last second it was seen)
    groups_by_t, level = [[] for _ in range(T)], [[0] * T for _ in range(n)]
    group_of = [[None] * T for _ in range(n)]
    for t in range(T):
        live = [i for i in range(n) if st[i][0][t] is not None and t - tracks[i]["first"] >= MIN_AGE]
        pos = {i: (st[i][0][t], st[i][1][t], st[i][2][t], st[i][3][t]) for i in live}
        for i in live:                                  # simultaneous-entry bookkeeping
            if i not in entered and math.hypot(pos[i][0], pos[i][1]) < ENTRY_R: entered[i] = t
        for i in live:
            a_ = aim(*pos[i]); ahist[i].append(a_[2] if a_ else None)
        steady_aim = {i: (len(ahist[i]) >= WINDOW and all(p is not None for p in ahist[i]) and
                          max(math.hypot(p[0] - q[0], p[1] - q[1]) for p in ahist[i] for q in ahist[i]) < AIM_DRIFT) for i in live}
        adj = {i: set() for i in live}; kinds = {}
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                i, j = live[a], live[b]
                dh = dhist.setdefault((i, j), deque(maxlen=WINDOW)); dh.append(math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]))
                steady = len(dh) >= WINDOW // 2 and _std(list(dh)) <= max(SPACING_SD, SPACING_FRAC * _mean(list(dh)))
                h = hist.setdefault((i, j), deque(maxlen=WINDOW)); h.append(link_type(pos[i], pos[j], steady, steady_aim[i], steady_aim[j]))
                got = [k for k in h if k]
                if len(got) >= PERSIST:
                    adj[i].add(j); adj[j].add(i); kinds[(i, j)] = max(set(got), key=got.count)
        for (i, j) in list(hist):                       # tracks that dropped out stop linking
            if (i not in pos or j not in pos) and hist[(i, j)]: hist[(i, j)].append(None)
        seen, comps = set(), []
        for i in live:
            if i in seen or not adj[i]: continue
            comp, stack = set(), [i]
            while stack:
                k = stack.pop()
                if k in comp: continue
                comp.add(k); stack.extend(adj[k] - comp)
            seen |= comp
            ks = {kinds.get((min(a, b), max(a, b))) for a in comp for b in comp if a < b} - {None}
            if len(comp) >= 3 or "shared target" in ks: comps.append((comp, ks))
        groups = []
        for comp, ks in comps:                          # keep group IDs stable: match the previous group sharing most members
            best = max(prev_groups, key=lambda g: len(g["members"] & comp), default=None)
            if best is not None and len(best["members"] & comp) > 0 and best["id"] not in [g["id"] for g in groups]:
                gid = best["id"]; split_merge = best["split_merge"] + (abs(len(best["members"]) - len(comp)) >= 2)
            else:
                gid = next_id; next_id += 1; split_merge = 0
            first_seen.setdefault(gid, t)
            m = sorted(comp)
            vx = sum(pos[i][2] for i in m) / len(m); vy = sum(pos[i][3] for i in m) / len(m)
            vel_spread = sum(math.hypot(pos[i][2] - vx, pos[i][3] - vy) for i in m) / len(m)
            dists = [math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) for ai, a in enumerate(m) for b in m[ai + 1:]]
            aims = [aim(*pos[i]) for i in m]; aims_ok = [a for a in aims if a and a[0] < AIM_R]
            etas = [a[1] for a in aims_ok]
            ent = sorted(entered[i] for i in m if i in entered)
            simult = any(ent[k + 1] - ent[k] <= ENTRY_DT for k in range(len(ent) - 1))
            onman = sum(tracks[i]["onman"][t] for i in m)
            recent = lambda i: any(tracks[i]["confn"][k] for k in range(max(0, t - RECENT_S), t + 1))     # confirmed within the last RECENT_S s
            confd = [i for i in m if recent(i)]
            birds = sum(1 for i in m if tracks[i]["cbird"][t])
            cen = math.hypot(sum(pos[i][0] for i in m) / len(m), sum(pos[i][1] for i in m) / len(m))
            unlisted_conf = [i for i in confd if not tracks[i]["onman"][t]]
            radio_silent = sum(1 for i in m if not tracks[i]["radio"][t]) / len(m)
            rmin = min(math.hypot(pos[i][0], pos[i][1]) for i in m)
            ch = chist.setdefault(gid, deque(maxlen=WINDOW)); ch.append(cen)
            closing = len(ch) >= WINDOW and (ch[0] - ch[-1]) / (len(ch) - 1) > 2.0      # group centre closing on the stadium at 2+ m/s
            converging = "shared target" in ks or closing
            unlisted = [i for i in m if not tracks[i]["onman"][t]]
            if onman >= 0.5 * len(m):
                # inside an approved group everything should be on the manifest: any member that is not is suspicious
                status = "intruder in approved group" if unlisted else "approved group"
            elif not confd or birds >= BIRD_SHARE * len(m):
                status = "coordinated, unconfirmed"      # nothing confirmed as a drone, or the camera sees mostly birds
            elif len(unlisted_conf) >= max(2, CONF_SHARE * len(m)):
                urgent = rmin <= URGENT_M or (etas and min(etas) <= URGENT_S)
                status = ("hostile swarm" if urgent else "hostile swarm (forming)") if converging else "suspicious formation"
            else:
                status = "coordinated group"
            old = status_hold.get(gid)                  # steps up at once; steps down only after the lower status has held for HOLD_DOWN s
            if old and STATUS_LEVEL[status] < STATUS_LEVEL[old[0]]:
                pend = old[1] if old[1] is not None else t
                if t - pend < HOLD_DOWN: status_hold[gid] = (old[0], pend); status = old[0]
                else: status_hold[gid] = (status, None)
            else:
                status_hold[gid] = (status, None)
            g = dict(id=gid, members=set(m), status=status, kinds=sorted(ks), size=len(m), split_merge=split_merge,
                     vel_spread=round(vel_spread, 1), spacing_cv=round((_std(dists) / max(_mean(dists), 1.0)) if len(dists) > 1 else 0.0, 2),
                     aim_spread=round(max((math.hypot(a[2][0] - b[2][0], a[2][1] - b[2][1]) for a in aims_ok for b in aims_ok), default=0.0)),
                     eta_spread=round(max(etas) - min(etas)) if len(etas) > 1 else None, eta_min=round(min(etas)) if etas else None,
                     simultaneous_entry=simult, persistence=t - first_seen[gid], radio_silent=round(radio_silent, 2),
                     on_manifest=onman, confirmed=len(confd), unlisted_confirmed=len(unlisted_conf), min_dist=round(rmin))
            groups.append(g)
            lv = STATUS_LEVEL[status]
            for i in m:
                group_of[i][t] = gid
                # approved groups never escalate their approved members; an intruder escalates only the intruder
                if status == "intruder in approved group" and tracks[i]["onman"][t]: continue
                level[i][t] = max(level[i][t], lv)
        # a group survives short gaps (noisy velocity while holding station) as long as its members are still tracked
        have = {g["id"] for g in groups}; taken = set().union(*[g["members"] for g in groups]) if groups else set()
        for gid, (g0, seen_t) in list(alive.items()):
            if gid in have: continue
            mem = {i for i in g0["members"] if i in pos and i not in taken}
            if t - seen_t <= GAP_S and len(mem) >= 2:
                g = dict(g0); g["members"] = mem; g["size"] = len(mem); g["persistence"] = t - first_seen[gid]
                groups.append(g); taken |= mem
                for i in mem:
                    group_of[i][t] = gid
                    if not (g["status"] == "intruder in approved group" and tracks[i]["onman"][t]): level[i][t] = max(level[i][t], STATUS_LEVEL[g["status"]])
            elif t - seen_t > GAP_S: del alive[gid]
        for g in groups:
            if g["id"] in have: alive[g["id"]] = (g, t)
        groups_by_t[t] = groups; prev_groups = groups
    return groups_by_t, level, group_of


def _mean(v): return sum(v) / len(v) if v else 0.0
def _std(v):
    if len(v) < 2: return 0.0
    m = _mean(v); return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
