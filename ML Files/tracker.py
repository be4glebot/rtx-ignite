"""Track management: IDs, coasting with growing uncertainty, and re-association.

The sensor model gives a fused position for each object in each second it is detected. A real tracker does
not know which object a detection belongs to, so this layer decides, like a real system would:
  * while detections keep coming, the track keeps its ID;
  * when they stop, the track COASTS: its position is predicted from its last velocity and the uncertainty
    ring grows (GATE0 + GROW m per second); after DROP_S seconds without a detection it is dropped;
  * when an object is detected again after a gap, the tracker looks for a coasting or recently dropped track
    whose predicted position is inside its uncertainty ring. Closest match (relative to ring size) wins:
      - the object's own old track  -> correct re-association (keeps its ID)
      - another object's track       -> identity swap (an error)
      - nothing inside the ring      -> a NEW ID (if the object had an ID before, it is now counted twice)
  * when two tracks pass close to each other, the tracker re-decides which is which when they separate,
    using their earlier velocities; similar velocities can cause a swap.
It also reports radar handoffs (which radar is tracking) and why a track was lost.
Ground truth is used only to score the decisions, never to make them.
"""
import math
import numpy as np

GATE0, GROW, GATE_MAX = 60.0, 7.0, 700.0   # uncertainty ring: start radius (m), growth (m/s), cap (m)
COAST_S = 3        # a gap longer than this is treated as a lost track that must be re-associated
DROP_S = 30        # a coasting track is dropped after this many seconds without a detection
CROSS_D = 40.0     # two tracks closer than this are "crossing"...
CROSS_DV = 4.0     # ...if their velocities differ by more than this (m/s); formation-mates flying together are not crossings
CROSS_LOOKBACK, CROSS_APART = 10, 150.0   # a crossing needs the two tracks to have been >150 m apart ~10 s earlier
RADAR_LOSE_S = 5   # a radar has lost a track after this many seconds without a detection


def _vel(fx, fy, k, back=6):
    """Velocity from the detections in the few seconds up to k (simple least-squares slope)."""
    ts = [j for j in range(max(0, k - back), k + 1) if fx[j] is not None and not math.isnan(fx[j])]
    if len(ts) < 2: return 0.0, 0.0
    t0 = sum(ts) / len(ts); den = sum((j - t0) ** 2 for j in ts) or 1.0
    vx = sum((j - t0) * fx[j] for j in ts) / den; vy = sum((j - t0) * fy[j] for j in ts) / den
    return vx, vy


def _smooth_vel(fx, fy, T):
    """Alpha-beta filter velocity per second (what a tracker carries), robust to single noisy fixes."""
    vx_, vy_ = [0.0] * T, [0.0] * T; x = None; y = vx = vy = 0.0; lastk = None
    for k in range(T):
        if fx[k] is None:
            if x is not None and lastk is not None and k - lastk > COAST_S: x = None
            vx_[k], vy_[k] = vx, vy; continue
        if x is None:
            x, y, vx, vy, lastk = fx[k], fy[k], 0.0, 0.0, k; vx_[k], vy_[k] = 0.0, 0.0; continue
        dt = k - lastk; x += vx * dt; y += vy * dt
        rx, ry = fx[k] - x, fy[k] - y; x += 0.35 * rx; y += 0.35 * ry; vx += 0.08 * rx / dt; vy += 0.08 * ry / dt; lastk = k
        vx_[k], vy_[k] = vx, vy
    return vx_, vy_


def associate(objs, T, pods, bowl, row=None):
    n = len(objs)
    fx = [[None if not np.isfinite(o["fused"][t, 0]) else float(o["fused"][t, 0]) for t in range(T)] for o in objs]
    fy = [[None if not np.isfinite(o["fused"][t, 1]) else float(o["fused"][t, 1]) for t in range(T)] for o in objs]
    det = [[fx[i][t] is not None for t in range(T)] for i in range(n)]
    sv = [_smooth_vel(fx[i], fy[i], T) for i in range(n)]
    V = lambda i, k: (sv[i][0][k], sv[i][1][k])
    did = [[0] * T for _ in range(n)]          # displayed track ID per object per second (0 = no track)
    state = [[0] * T for _ in range(n)]        # 0 none, 1 tracked, 2 coasting (estimated)
    unc = [[0] * T for _ in range(n)]          # uncertainty ring radius (m)
    ex = [[None] * T for _ in range(n)]; ey = [[None] * T for _ in range(n)]   # estimated position while coasting
    seg_start = [[None] * T for _ in range(n)]
    doubt = [[0] * T for _ in range(n)]        # "possible duplicate": a new track started near a track that is still coasting
    events = []
    tracks = {}       # id -> dict(obj=true owner, last_t, x, y, vx, vy, start, alive)
    holder = [None] * n   # current track id held by each object
    ids_of = [set() for _ in range(n)]
    next_id = 1
    stats = dict(reacq=0, correct=0, duplicates=0, swaps=0, mate_swaps=0, new_after_loss=0, crossings=0, cross_swaps=0, handoffs=0, radar_gaps=0)
    close_pairs = {}
    for t in range(T):
        # 1) objects detected this second
        for i in range(n):
            if not det[i][t]: continue
            h = holder[i]
            if h is not None and tracks[h]["alive"] and t - tracks[h]["last_t"] <= COAST_S:
                tr = tracks[h]; tr["last_t"] = t; tr["x"], tr["y"] = fx[i][t], fy[i][t]; tr["vx"], tr["vy"] = V(i, t)
                continue
            # new detection after a gap (or first ever): look for a coasting / recently dropped track inside its ring
            best, best_r, best_info = None, 1e9, None
            for tid, tr in tracks.items():
                if tr["last_t"] >= t or t - tr["last_t"] > DROP_S: continue
                if any(holder[j] == tid and det[j][t] and j != i and tracks[tid]["last_t"] == t for j in range(n)): continue
                gap = t - tr["last_t"]; px, py = tr["x"] + tr["vx"] * gap, tr["y"] + tr["vy"] * gap
                ring = min(GATE0 + GROW * gap, GATE_MAX); d = math.hypot(fx[i][t] - px, fy[i][t] - py)
                if d <= ring and d / ring < best_r: best, best_r, best_info = tid, d / ring, (d, ring, gap)
            had_id = holder[i] is not None
            if best is not None and not any(holder[j] == best and det[j][t] for j in range(n) if j != i):
                tr = tracks[best]; owner = tr["obj"]
                for j in range(n):            # the track moves to this object
                    if holder[j] == best and j != i: holder[j] = None
                holder[i] = best; tr["obj"] = i; tr["alive"] = True; tr["last_t"] = t
                tr["x"], tr["y"] = fx[i][t], fy[i][t]; tr["vx"], tr["vy"] = V(i, t)
                ok = owner == i; stats["reacq"] += 1; stats["correct"] += ok
                mates = (not ok) and _formation_mates(objs[i], objs[owner], t)
                stats["swaps"] += (not ok) and not mates; stats["mate_swaps"] += mates
                d, ring, gap = best_info
                events.append([t, "Track %d reacquired after %d s: matched (%.0f m from its predicted position, inside the %.0f m ring)%s"
                               % (best, gap, d, ring, "" if ok else (" [scoring: a formation-mate, identical drones swapped]" if mates else " [scoring: WRONG drone, identities swapped]"))])
                seg_start[i][t] = tr["start"]
            else:
                tid = next_id; next_id += 1
                tracks[tid] = dict(obj=i, last_t=t, x=fx[i][t], y=fy[i][t], vx=0.0, vy=0.0, start=t, alive=True)
                near = [k for k, tr in tracks.items() if k != tid and tr["alive"] and 0 < t - tr["last_t"] <= DROP_S and
                        math.hypot(fx[i][t] - (tr["x"] + tr["vx"] * (t - tr["last_t"])), fy[i][t] - (tr["y"] + tr["vy"] * (t - tr["last_t"]))) <= 2.5 * min(GATE0 + GROW * (t - tr["last_t"]), GATE_MAX)]
                if near:
                    for k in range(t, min(T, t + 15)): doubt[i][k] = 1
                    events.append([t, "Track %d may be the same drone as lost track %d (just outside its ring): flagged for human review" % (tid, near[0])])
                if had_id:
                    stats["new_after_loss"] += 1
                    old = holder[i]
                    events.append([t, "New track %d started: no lost track inside its uncertainty ring [scoring: same drone as track %d, now counted twice]" % (tid, old)])
                ids_of[i].add(tid); holder[i] = tid
        # 2) crossing tracks: when two tracked objects separate after passing close, decide who is who
        act = [i for i in range(n) if det[i][t] and holder[i] is not None]
        for a in range(len(act)):
            for b in range(a + 1, len(act)):
                i, j = act[a], act[b]; key = (i, j)
                d = math.hypot(fx[i][t] - fx[j][t], fy[i][t] - fy[j][t])
                vi_, vj_ = V(i, t), V(j, t)
                tb = next((k for k in range(t - CROSS_LOOKBACK, max(t - 3 * CROSS_LOOKBACK, -1), -1) if k >= 0 and det[i][k] and det[j][k]), None)
                came_from_apart = tb is not None and math.hypot(fx[i][tb] - fx[j][tb], fy[i][tb] - fy[j][tb]) > CROSS_APART
                if d < CROSS_D and key not in close_pairs and came_from_apart and math.hypot(vi_[0] - vj_[0], vi_[1] - vj_[1]) > CROSS_DV:   # paths actually crossing
                    # remember where each was heading BEFORE they met (average over the 5 s before), to re-pair them after
                    avg = lambda q: tuple(sum(V(q, k)[c] for k in range(max(0, t - 6), max(t - 1, 1))) / max(1, max(t - 1, 1) - max(0, t - 6)) for c in (0, 1))
                    close_pairs[key] = (t, avg(i), avg(j), fx[i][t], fy[i][t], fx[j][t], fy[j][t])
                elif d >= CROSS_APART and key in close_pairs and t - close_pairs[key][0] >= 3:
                    t0, vi, vj, xi, yi, xj, yj = close_pairs.pop(key); dt = t - t0
                    pi = (xi + vi[0] * dt, yi + vi[1] * dt); pj = (xj + vj[0] * dt, yj + vj[1] * dt)
                    keep = math.hypot(fx[i][t] - pi[0], fy[i][t] - pi[1]) + math.hypot(fx[j][t] - pj[0], fy[j][t] - pj[1])
                    swap = math.hypot(fx[i][t] - pj[0], fy[i][t] - pj[1]) + math.hypot(fx[j][t] - pi[0], fy[j][t] - pi[1])
                    stats["crossings"] += 1
                    hi, hj = holder[i], holder[j]
                    if swap < keep:
                        holder[i], holder[j] = hj, hi; tracks[hj]["obj"] = i; tracks[hi]["obj"] = j; stats["cross_swaps"] += 1
                        events.append([t, "Tracks %d and %d crossed: tracker re-paired them by heading [scoring: identities SWAPPED]" % (hi, hj)])
                    else:
                        events.append([t, "Tracks %d and %d crossed: identities kept (re-paired by heading)" % (hi, hj)])
        # 3) record state per object; coasting tracks get an estimated position and a growing ring
        for i in range(n):
            h = holder[i]
            if h is None: continue
            tr = tracks[h]
            if not tr["alive"]: continue
            gap = t - tr["last_t"]
            if gap > DROP_S:
                tr["alive"] = False
                events.append([t, "Track %d dropped after %d s without a detection" % (h, DROP_S)]); continue
            did[i][t] = h; seg_start[i][t] = seg_start[i][t] or tr["start"]
            if gap == 0:
                state[i][t] = 1; unc[i][t] = int(GATE0 * 0.5)
            else:
                state[i][t] = 2; unc[i][t] = int(min(GATE0 + GROW * gap, GATE_MAX))
                ex[i][t] = tr["x"] + tr["vx"] * gap; ey[i][t] = tr["y"] + tr["vy"] * gap
                if gap == COAST_S + 1:
                    events.append([t - gap + 1, "Track %d lost %s: estimating its position, uncertainty growing" % (h, _why_lost(objs[i], t - gap + 1, pods, bowl))])
    # duplicates: objects that ended up with more than one ID
    stats["duplicates"] = sum(max(0, len(s) - 1) for s in ids_of)
    # radar handoff: a radar "holds" a track while it has detected it in the last RADAR_LOSE_S seconds.
    # The tracking radar only changes when it has lost the target and the other radar holds it.
    radar_active = [[0] * T for _ in range(n)]
    names = list(pods)
    for i, o in enumerate(objs):
        rb = o.get("rad_by", {})
        last = {r: -99 for r in rb}; cur = None; no_radar_since = None
        for t in range(T):
            for r in rb:
                if rb[r][t]: last[r] = t
            holding = [r for r in rb if t - last[r] < RADAR_LOSE_S]
            mask = 0
            for r in holding: mask |= 1 << names.index(r)
            radar_active[i][t] = mask
            if cur is not None and cur not in holding:
                if holding and did[i][t]:
                    new = holding[0]; stats["handoffs"] += 1
                    events.append([t, "Radar %s takes over track %d from radar %s (%s); the other radar already had it, so no gap" % (new, did[i][t], cur, _why_radar(o, cur, last[cur] + 1, pods, bowl))])
                    cur = new
                elif not holding and no_radar_since is None:
                    no_radar_since = t
                    if did[i][t]: events.append([t, "Radar lost track %d (%s); cameras and RF keep it if they can" % (did[i][t], _why_radar(o, cur, last[cur] + 1, pods, bowl))])
            if holding and (cur is None or cur not in holding):
                if no_radar_since is not None and did[i][t]:
                    stats["radar_gaps"] += 1
                    events.append([t, "Radar %s picks track %d up again after %d s without radar" % (holding[0], did[i][t], t - no_radar_since)])
                cur = holding[0]; no_radar_since = None
    events.sort(key=lambda e: e[0])
    return dict(doubt=doubt, did=did, state=state, unc=unc, ex=ex, ey=ey, seg_start=seg_start, radar=radar_active, events=events, stats=stats)


def _formation_mates(a, b, t):
    """Scoring only: two drones flying together (close, same velocity) are interchangeable members of one formation."""
    pa, pb = a["truth"][t], b["truth"][t]
    if not (np.isfinite(pa[0]) and np.isfinite(pb[0]) and t > 0): return False
    va = a["truth"][t] - a["truth"][t - 1]; vb = b["truth"][t] - b["truth"][t - 1]
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 120 and np.nan_to_num(np.hypot(va[0] - vb[0], va[1] - vb[1]), nan=99) < 3


def _why_lost(o, t, pods, bowl):
    x, y, z = o["truth"][t] if t < len(o["truth"]) else (np.nan, np.nan, np.nan)
    if not np.isfinite(x): return "(left coverage or landed)"
    from simulate import line_of_sight, SITE_Z
    if abs(x) < bowl[0] and abs(y) < bowl[1] and all(not line_of_sight(np.array(pods[r]), SITE_Z[r]["radar"], np.array([[x, y]]), np.array([z]))[0] for r in o.get("rad_by", {})):
        return "low over the bowl, below the stands' rim as seen from both radars"
    if not o.get("rad_by"): return "(no radar available to see it)"
    if all(not line_of_sight(np.array(pods[r]), SITE_Z[r]["radar"], np.array([[x, y]]), np.array([z]))[0] for r in o.get("rad_by", {})):
        return "hidden behind the stadium or buildings from both radars"
    return "(detection gap)"


def _why_radar(o, r, t, pods, bowl):
    from simulate import line_of_sight, SITE_Z, RADAR_RANGE
    x, y, z = o["truth"][t]
    if not np.isfinite(x): return "target left"
    if math.hypot(x - pods[r][0], y - pods[r][1]) > RADAR_RANGE: return "out of its range"
    if not line_of_sight(np.array(pods[r]), SITE_Z[r]["radar"], np.array([[x, y]]), np.array([z]))[0]: return "its view is blocked by the stadium or a building"
    return "it lost the target"
