"""Regenerate the 10 demo scenarios from designed flight paths.
Every field the page reads is derived here from the paths with one consistent set of rules."""
import json, math, random, re, base64, gzip, sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
m = re.search(r'const DZ\s*=\s*"([^"]+)"', s)
D = json.loads(gzip.decompress(base64.b64decode(m.group(1))))
OLD = {c['scenario_id']: c for c in D['cases']}
T = D['T']
PODS = list(D['pods'].keys())          # NE NW SE SW CCG WCG
PXY = D['pods']
RADARS = D['radars']
BX, BY = D['bowl']
LABELS = D['labels']
LI = {l: i for i, l in enumerate(LABELS)}
CAM_RANGE = 800

# ---------------- path building ----------------
class Path:
    def __init__(self, x, y, z, t0=0):
        self.t0 = t0; self.p = [(x, y, z)]
    def go(self, x, y, z, v):
        x0, y0, z0 = self.p[-1]
        d = math.dist((x0, y0, z0), (x, y, z)); n = max(1, round(d / v))
        for i in range(1, n + 1):
            f = i / n; self.p.append((x0 + (x - x0) * f, y0 + (y - y0) * f, z0 + (z - z0) * f))
        return self
    def hold(self, secs, wob=2.0, seed=0):
        x0, y0, z0 = self.p[-1]; r = random.Random(seed)
        ph = r.uniform(0, 6.28)
        for i in range(1, secs + 1):
            self.p.append((x0 + wob * math.sin(ph + i * 0.21), y0 + wob * math.cos(ph + i * 0.17), z0 + 0.5 * math.sin(i * 0.3)))
        return self
    def orbit(self, cx, cy, r, secs, v, z=None, ccw=True):
        x0, y0, z0 = self.p[-1]; a = math.atan2(y0 - cy, x0 - cx); w = v / r * (1 if ccw else -1)
        zz = z0 if z is None else z
        for i in range(1, secs + 1):
            self.p.append((cx + r * math.cos(a + w * i), cy + r * math.sin(a + w * i), zz))
        return self
    def end(self):
        out = [None] * T
        for i, q in enumerate(self.p):
            k = self.t0 + i
            if 0 <= k < T: out[k] = q
        return out

def smoothpath(arr, w=3):
    """light moving average so turns are rounded, keeps None gaps"""
    out = list(arr)
    for k in range(T):
        if arr[k] is None: continue
        pts = [arr[j] for j in range(max(0, k - w), min(T, k + w + 1)) if arr[j] is not None]
        out[k] = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
    return out

# ---------------- scenarios ----------------
def sc_broadcast():
    P = Path(-30, -40, 65)
    for i, (x, y) in enumerate([(25, -45), (25, 10), (-20, 30), (-25, -10), (20, 45), (0, 0), (-30, -40), (25, -20), (10, 40)] * 2):
        P.go(x, y, 62 + 6 * math.sin(i), 2.5).hold(6, 1.5, i)
    return [dict(path=smoothpath(P.end()), kind='multirotor_drone', script=['authorized'], rf=1, rid=1)]

def sc_birds():
    out = []; r = random.Random(9)
    for i in range(12):
        ox, oy = r.uniform(-110, 110), r.uniform(-70, 70); z = r.uniform(35, 60)
        pts = [None] * T
        for k in range(T):
            x = -1400 + 12 * k + ox + 6 * math.sin(k * 0.13 + i)
            y = 120 - 0.9 * k * 0.3 + oy + 5 * math.cos(k * 0.11 + i * 1.7)
            pts[k] = (x, y, z + 3 * math.sin(k * 0.2 + i)) if abs(x) < 1500 else None
        out.append(dict(path=pts, kind='bird', script=['non_drone'], rf=0, rid=0))
    return out

def sc_hobbyist():
    P = Path(520, -480, 5, t0=8).go(520, -480, 40, 3).go(150, -150, 80, 8).go(60, -40, 80, 6)
    P.orbit(0, 0, 70, 90, 5).go(180, -200, 70, 7).go(560, -520, 30, 8).go(560, -520, 3, 3)
    return [dict(path=smoothpath(P.end()), kind='multirotor_drone', script=['unauthorized_careless'], rf=1, rid=0, att=2)]

def sc_delivery():
    P = Path(-1500, -700, 70).go(-700, -700, 70, 14)
    P.go(-250, -380, 70, 13).go(100, -330, 70, 12).go(500, -650, 70, 13).go(1500, -720, 70, 14)
    pts = smoothpath(P.end(), 4)
    def lab(k, q):   # authorized while on the approved east-west route, borderline once it drifts off
        return 'authorized' if abs(q[1] + 700) < 60 and k < 120 else 'borderline'
    return [dict(path=pts, kind='multirotor_drone', script=lab, rf=1, rid=1, approved=True)]

def sc_probing():
    P = Path(1150, 1150, 60, t0=4).go(380, 380, 60, 12).hold(14, 3, 5).go(1250, 1150, 70, 13)
    return [dict(path=smoothpath(P.end()), kind='multirotor_drone', script=['borderline'], rf=1, rid=0)]

def sc_radiosilent():
    P = Path(-1300, 1300, 60, t0=5).go(-60, 60, 55, 21).go(0, 0, 20, 8)
    return [dict(path=smoothpath(P.end(), 2), kind='multirotor_drone', script=['threat'], rf=0, rid=0)]

def sc_payload():
    P = Path(-950, -150, 50, t0=3).go(-200, -40, 50, 10).go(95, 10, 45, 6).hold(150, 2.5, 3)
    return [dict(path=smoothpath(P.end()), kind='multirotor_drone', script=['threat'], rf=1, rid=0, att=3)]

def sc_hijack():
    cx, cy = -420, 380
    P = Path(cx + 60, cy, 50).orbit(cx, cy, 60, 40, 6).go(-120, 110, 50, 15).go(0, 0, 40, 12).hold(20, 2, 8)
    pts = smoothpath(P.end())
    def lab(k, q): return 'authorized' if k < 44 else 'threat'
    return [dict(path=pts, kind='multirotor_drone', script=lab, rf=1, rid=1, approved=True)]

def sc_swarm():
    out = []
    targets = [(-70, 40), (70, 40), (0, -60)]          # three stands
    for i in range(12):
        row, col = divmod(i, 4)
        ox, oy = (col - 1.5) * 45, row * 45
        grp = col % 3 if col < 3 else (row % 3)
        P = Path(ox, 1650 + oy, 82, t0=0).go(ox, 420 + oy, 82, 11)
        tx, ty = targets[grp]
        P.go(tx + ox * 0.3, ty + 150 + oy * 0.3, 60, 12).go(tx + ox * 0.25, ty + oy * 0.2, 25, 10).hold(30, 1.5, i)
        out.append(dict(path=smoothpath(P.end(), 2), kind='multirotor_drone', script=['threat'], rf=1, rid=0, swarm=True))
    return out

def sc_lightshow():
    out = []
    for i in range(10):
        pts = [None] * T
        for k in range(T):
            a = i / 10 * 6.283 + k * 0.035
            r = 55 + 15 * math.sin(k * 0.05)
            pts[k] = (r * math.cos(a), r * math.sin(a), 75 + 15 * math.sin(k * 0.06 + i))
        out.append(dict(path=pts, kind='multirotor_drone', script=['authorized'], rf=1, rid=1, bg=1, approved=True, show=True))
    P = Path(60, -700, 70, t0=10).go(30, -120, 75, 8)
    k0 = 10 + len(P.p) - 1
    pts = P.end()
    for k in range(k0 + 1, T):   # slips into the ring and rides along with it
        a = 0.25 + k * 0.035; r = 55 + 15 * math.sin(k * 0.05)
        pts[k] = (r * 1.15 * math.cos(a), r * 1.15 * math.sin(a), 80 + 10 * math.sin(k * 0.06))
    out.append(dict(path=smoothpath(pts, 3), kind='multirotor_drone', script=['threat'], rf=1, rid=0, show=True, intruder=True))
    return out

SCEN = [(3, sc_broadcast), (9, sc_birds), (18, sc_hobbyist), (26, sc_delivery), (32, sc_probing),
        (39, sc_radiosilent), (42, sc_payload), (80, sc_hijack), (52, sc_swarm), (76, sc_lightshow)]

# ---------------- sensor + system model ----------------
def edge_dist(x, y):
    r = math.hypot(x, y)
    if r < 1: return 0
    th = math.atan2(y, x); R = 1 / math.sqrt((math.cos(th) / BX) ** 2 + (math.sin(th) / BY) ** 2)
    return max(0, r - R)

def radar_sees(q):
    x, y, z = q
    if math.hypot(x, y) < 150 and z < 35: return False      # low over the bowl: stands block both radars
    return z > 8 and any(math.hypot(x - PXY[r][0], y - PXY[r][1]) < 2600 for r in RADARS)

PRI = {'threat': 5, 'borderline': 4, 'unauthorized_careless': 3, None: 2, 'authorized': 1, 'non_drone': 0}

def build_run(objs, key, meta):
    no_r, no_f, no_c = 'no_radar' in key, 'no_rf' in key, 'no_cam' in key
    rnd = random.Random(hash(key) & 0xffff)
    n = len(objs)
    TR = []
    for i, o in enumerate(objs):
        pth = o['path']
        TR.append(dict(o=o, det=[0] * T, rad=[0] * T, lock=[0] * T, asg=[0] * T, cid=[0] * T))
    # raw sensor detections (before cameras)
    for tr in TR:
        o = tr['o']
        for k in range(T):
            q = o['path'][k]
            if q is None: continue
            d = 0
            if not no_r and radar_sees(q):
                d |= 1; tr['rad'][k] = sum(1 << PODS.index(r) for r in RADARS)
            if not no_f and o['rf'] and min(math.hypot(q[0] - PXY[p][0], q[1] - PXY[p][1]) for p in PODS) < 2000: d |= 2
            if not no_f and o['rid'] and min(math.hypot(q[0] - PXY[p][0], q[1] - PXY[p][1]) for p in PODS) < 1500: d |= 4
            tr['det'][k] = d
    # track life: alive while a sensor (or a locked camera) has it, coast up to 8 s after
    def label_at(tr, k):
        o = tr['o']; q = o['path'][k]
        if q is None: return None
        sc = o['script']
        return sc(k, q) if callable(sc) else sc[0]
    # cameras: greedy, sticky, one target each; needs a cue (radar/RF) or an earlier lock
    cams = [] if no_c else PODS[:]
    cur = {p: None for p in cams}; since = {p: 0 for p in cams}
    known = [False] * n
    for k in range(T):
        cued = []
        for i, tr in enumerate(TR):
            q = tr['o']['path'][k]
            if q is None: continue
            if tr['det'][k] or (k and tr['lock'][k - 1]) or known[i]:
                cued.append(i)
        taken = set()
        for p in cams:
            c = cur[p]
            q = TR[c]['o']['path'][k] if c is not None else None
            ok = c is not None and q is not None and c in cued and math.hypot(q[0] - PXY[p][0], q[1] - PXY[p][1]) <= CAM_RANGE
            # give up an identified, low-priority target if an unidentified one is waiting
            if ok and TR[c]['cid'][k - 1 if k else 0]:
                waiting = [i for i in cued if i not in taken and not TR[i]['cid'][k - 1 if k else 0] and i not in cur.values()
                           and math.hypot(TR[i]['o']['path'][k][0] - PXY[p][0], TR[i]['o']['path'][k][1] - PXY[p][1]) <= CAM_RANGE]
                if waiting and k - since[p] > 8: ok = False
            if not ok or c in taken:
                best = None
                for i in cued:
                    if i in taken or i in cur.values() and cur.get(p) != i: continue
                    q2 = TR[i]['o']['path'][k]
                    dd = math.hypot(q2[0] - PXY[p][0], q2[1] - PXY[p][1])
                    if dd > CAM_RANGE: continue
                    sc = (1 if not TR[i]['cid'][k - 1 if k else 0] else 0, PRI[label_at(TR[i], k)], -dd)
                    if best is None or sc > best[0]: best = (sc, i)
                newc = best[1] if best else None
                if newc != c: since[p] = k
                cur[p] = newc
            if cur[p] is not None:
                taken.add(cur[p]); i = cur[p]; bit = 1 << PODS.index(p)
                TR[i]['asg'][k] |= bit
                if k - since[p] >= 2: TR[i]['lock'][k] |= bit
        for i, tr in enumerate(TR):
            if tr['lock'][k]:
                tr['det'][k] |= 8
                lk = sum(1 for j in range(max(0, k - 2), k + 1) if tr['lock'][j])
                if lk >= 2 or tr['cid'][k - 1 if k else 0]: known[i] = True
            if known[i] and tr['o']['path'][k] is not None:
                tr['cid'][k] = {'multirotor_drone': 1, 'bird': 2}.get(tr['o']['kind'], 3)
    # assemble tracks
    tracks = []; nid = 0; ev = []
    for i, tr in enumerate(TR):
        o = tr['o']; pth = o['path']
        first = next((k for k in range(T) if tr['det'][k]), None)
        if first is None:
            first = T - 1
        nid += 1
        did = [0] * T; st = [0] * T; unc = [30] * T; fx = [None] * T; fy = [None] * T; ex = [None] * T; ey = [None] * T
        lastk = None; lastv = (0, 0); nx = ny = 0.0
        for k in range(first, T):
            q = pth[k]
            nx = 0.7 * nx + rnd.gauss(0, 3.5); ny = 0.7 * ny + rnd.gauss(0, 3.5)
            if q is None:
                continue
            if tr['det'][k]:
                sig = 1.0 if tr['det'][k] & 9 else 4.0      # RF-only fixes are much rougher
                fx[k] = round(q[0] + nx * sig); fy[k] = round(q[1] + ny * sig)
                if lastk is not None and k - lastk > 1 and st[k - 1] == 2:
                    ev.append([k, f'Track {nid} reacquired'])
                did[k] = nid; st[k] = 1; lastk = k
                if k > 0 and pth[k - 1] is not None: lastv = (q[0] - pth[k - 1][0], q[1] - pth[k - 1][1])
            elif lastk is not None and k - lastk <= 8:
                dt = k - lastk; did[k] = nid; st[k] = 2; unc[k] = 30 + 12 * dt
                lq = pth[lastk]; ex[k] = round(lq[0] + lastv[0] * dt); ey[k] = round(lq[1] + lastv[1] * dt)
                if dt == 1: ev.append([k, f'Track {nid} lost by every sensor: estimating its position, uncertainty growing'])
        for k in range(first + 1, T):
            if did[k] and st[k] == 1 and (tr['det'][k - 1] & 1) and not (tr['det'][k] & 1) and pth[k] is not None and math.hypot(pth[k][0], pth[k][1]) < 160:
                ev.append([k, f'Radar lost track {nid} (low over the bowl, the stands block it); cameras and RF keep it'])
        # categories, alert levels, time to bowl
        pred = [-1] * T; lvl = [0] * T; ttb = [None] * T
        for k in range(T):
            q = pth[k]
            if q is None: continue
            d = edge_dist(q[0], q[1])
            k0 = max(0, k - 3)
            while k0 < k and pth[k0] is None: k0 += 1
            if d == 0: ttb[k] = 0
            elif k0 < k:
                cl = (math.hypot(*pth[k0][:2]) - math.hypot(*q[:2])) / (k - k0)
                if cl > 1 and d / cl < 600: ttb[k] = round(d / cl)
            if not did[k] or k < first + 3: continue
            lab = label_at(tr, k); pred[k] = LI[lab]
            if lab in ('unauthorized_careless', 'borderline'): lvl[k] = 2
            elif lab == 'threat':
                urgent = math.hypot(q[0], q[1]) < 600 or d == 0 or (ttb[k] is not None and ttb[k] <= 120)
                confirmed = tr['cid'][k] == 1 or sum(1 for j in range(max(0, k - 10), k + 1) if tr['det'][j] & 1) >= 6 or (tr['det'][k] & 2)
                lvl[k] = 3 if urgent and confirmed else 2
        att = [0] * T; attc = [0] * T
        for k in range(T):
            if tr['lock'][k]: att[k] = o.get('att', 1); attc[k] = 20
        tracks.append(dict(bg=o.get('bg', 0), kind=o['kind'], first=first,
            tx=[None if q is None else round(q[0]) for q in pth], ty=[None if q is None else round(q[1]) for q in pth],
            tz=[None if q is None else round(q[2]) for q in pth], fx=fx, fy=fy, ex=ex, ey=ey, st=st, unc=unc, did=did,
            det=tr['det'], rad=tr['rad'], lock=tr['lock'], asg=tr['asg'], cid=tr['cid'], att=att, attc=attc,
            pred=pred, lvl=lvl, slvl=[0] * T, grp=[0] * T, ttb=ttb, conf=[80] * T, rev=[0] * T,
            cam={'multirotor_drone': 'drone', 'bird': 'bird'}.get(o['kind'], 'other')))
    # groups
    groups = [[] for _ in range(T)]
    def members(filt, k): return [i for i, t in enumerate(tracks) if filt(i) and t['did'][k] and t['tx'][k] is not None]
    gstart = None
    for k in range(T):
        sid = meta['scenario_id']
        if sid == 9: mem = members(lambda i: True, k)
        elif sid == 52: mem = members(lambda i: True, k)
        elif sid == 76: mem = members(lambda i: objs[i].get('show') and (not objs[i].get('intruder') or math.hypot(tracks[i]['tx'][k] or 9e9, tracks[i]['ty'][k] or 9e9) < 160), k)
        else: mem = []
        if len(mem) < 3: gstart = None; continue
        if gstart is None: gstart = k
        if k - gstart < 5: continue      # a link has to hold a few seconds
        pts = [(tracks[i]['tx'][k], tracks[i]['ty'][k]) for i in mem]
        vel = []
        for i in mem:
            a, b = tracks[i]['tx'], tracks[i]['ty']
            if k and a[k - 1] is not None: vel.append((a[k] - a[k - 1], b[k] - b[k - 1]))
        mvx = sum(v[0] for v in vel) / max(1, len(vel)); mvy = sum(v[1] for v in vel) / max(1, len(vel))
        vs = round(sum(math.hypot(v[0] - mvx, v[1] - mvy) for v in vel) / max(1, len(vel)), 1)
        cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
        ds = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]; md = sum(ds) / len(ds)
        cv = round((sum((d - md) ** 2 for d in ds) / len(ds)) ** 0.5 / max(md, 1), 2)
        conf = sum(1 for i in mem if tracks[i]['cid'][k] == 1)
        birds = sum(1 for i in mem if tracks[i]['cid'][k] == 2)
        onm = sum(1 for i in mem if objs[i].get('approved'))
        if sid == 9: st = 'bird flock' if birds >= 0.3 * len(mem) else 'coordinated, unconfirmed'
        elif sid == 52:
            st = 'hostile swarm' if (conf >= 2 and math.hypot(cx, cy) < 1000) else ('suspicious formation' if conf >= 2 else 'coordinated, unconfirmed')
        else: st = 'intruder in approved group' if onm < len(mem) else 'approved group'
        silent = sum(1 for i in mem if not objs[i]['rf']) / len(mem)
        kinds = 'holding' if math.hypot(mvx, mvy) < 2 else 'co-moving'
        sm = 1 if sid == 52 and cy < 380 else 0
        groups[k].append([1, st, mem, len(mem), vs, cv, 0, None, 1, k - gstart, round(silent, 2), onm, conf, kinds, sm])
        for i in mem:
            tracks[i]['grp'][k] = 1
            if st == 'hostile swarm': tracks[i]['slvl'][k] = 3
            elif st == 'suspicious formation': tracks[i]['slvl'][k] = 2
            elif st == 'intruder in approved group' and not objs[i].get('approved'): tracks[i]['slvl'][k] = 2
    return dict(tracks=tracks, groups=groups, tev=sorted(ev), cams=cams, cam_range=CAM_RANGE, assoc={})

KEYS = ['', 'no_radar', 'no_rf', 'no_cam', 'no_radar+no_rf', 'no_radar+no_cam', 'no_rf+no_cam']
cases = []
for sid, fn in SCEN:
    o = OLD[sid]
    meta = {k: v for k, v in o.items() if k != 'runs'}
    objs = fn()
    meta['n_true'] = sum(1 for x in objs if not x.get('bg'))
    meta['rf_link'] = 'True' if any(x['rf'] for x in objs if not x.get('bg')) else 'False'
    meta['remote_id'] = 'valid' if all(x['rid'] for x in objs if not x.get('bg')) else 'none'
    meta['on_list'] = 'True' if all(x.get('approved') for x in objs if not x.get('bg')) else 'False'
    if sid == 9: meta['on_list'] = 'False'
    meta['runs'] = {k: build_run(objs, k, meta) for k in KEYS}
    cases.append(meta)
D['cases'] = cases; D['pool'] = []
json.dump(D, open('/tmp/fix2/G.json', 'w'))
blob = base64.b64encode(gzip.compress(json.dumps(D, separators=(',', ':')).encode(), 9)).decode()
s = s[:m.start(1)] + blob + s[m.end(1):]
open(dst, 'w').write(s)
print('ok', len(s))
