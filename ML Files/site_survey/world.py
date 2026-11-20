"""Kyle Field surroundings: a 2 m height map from OpenStreetMap buildings plus our FreeCAD stadium, and radar line-of-sight on it.
Frame: x = east, y = north (metres), origin at the OSM Kyle Field centroid. Flat ground assumed. Trees are NOT included."""
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent          # this folder (site_survey/)
from matplotlib.path import Path
o = json.load(open(HERE / "origin.json")); LAT0, LON0 = o["lat0"], o["lon0"]
def en(c): c = np.asarray(c, float); return np.c_[(c[:, 0]-LON0)*111320*np.cos(np.radians(LAT0)), (c[:, 1]-LAT0)*110574]
FIELD_BRG = 321.0                                  # field long axis (north end), from the campus grid; MSC lies north, Bright Complex south
N_HAT = np.array([np.sin(np.radians(FIELD_BRG)), np.cos(np.radians(FIELD_BRG))])
W_HAT = np.array([np.sin(np.radians(FIELD_BRG-90)), np.cos(np.radians(FIELD_BRG-90))])   # CAD +y = west side (right-handed, x north)
CAD_ORIGIN = np.array([6.4, 2.0])                  # OSM footprint centroid
def cad2en(x, y): return CAD_ORIGIN + np.outer(x, N_HAT) + np.outer(y, W_HAT)
# ---- building heights (OSM has floor counts for some; everything else is an estimate) ----
PER_LEVEL = 3.6
OVERRIDE = {  # name: (height m, note)  -- estimates, to be checked
    "Rudder Tower": (46, "11 floors in OSM; ~46 m est."), "Texas A&M Hotel and Conference Center": (30, "est. ~8 floors"),
    "Memorial Student Center": (18, "est."), "Koldus Building": (18, "est."), "Bright Football Complex": (18, "est."),
    "McFerrin Athletic Center-Indoor Track": (24, "est."), "McFerrin Athletic Center-Indoor Football": (32, "est. indoor practice field"),
    "Student Recreation Center": (16, "est."), "Reed Arena": (30, "est."), "University Center Garage": (16, "est. 5 decks"),
    "Gene Stallings Blvd Garage": (18, "est. 5-6 decks"), "West Campus Garage": (23.9, "7 levels (user-confirmed)"), "Southside Parking Garage": (16, "est."),
    "Albritton Bell Tower": (42, "est."), "Interdisciplinary Life Sciences Building": (22, "est."), "Duncan Dining Hall": (12, "est.")}
DEFAULT = {"university": 14, "yes": 10, "apartments": 13, "dormitory": 16, "residential": 12, "detached": 7, "house": 7, "retail": 6,
           "roof": 5, "industrial": 10, "church": 12, "office": 15, "parking": 14, "stadium": 12, "commercial": 8, "bridge": 0}
def building_height(p):
    n = p.get("name")
    if n in OVERRIDE: return OVERRIDE[n][0]
    if p.get("man_made") == "water_tower": return 40
    if p.get("man_made") == "tower": return 30
    if p.get("building:levels"):
        try: lv = float(p["building:levels"]); return lv*(3.2 if p.get("building") == "parking" else PER_LEVEL) + 1.5
        except ValueError: pass
    return DEFAULT.get(p.get("building"), 10)
CELL = 2.0; EXT = 1700.0
NX = int(2*EXT/CELL)
def load():
    d = json.load(open(HERE / "kf.geojson")); H = np.zeros((NX, NX), np.float32); blds = []
    gx = (np.arange(NX)+0.5)*CELL - EXT
    for f in d["features"]:
        g, p = f["geometry"], f["properties"]
        if g["type"] != "Polygon" or p.get("name") == "Kyle Field": continue
        ring = en(g["coordinates"][0]); h = building_height(p)
        if h <= 0: continue
        x0, y0 = ring.min(0); x1, y1 = ring.max(0)
        i0, i1 = np.searchsorted(gx, [x0, x1]); j0, j1 = np.searchsorted(gx, [y0, y1])
        if i1 <= i0 or j1 <= j0: continue
        XX, YY = np.meshgrid(gx[i0:i1], gx[j0:j1], indexing="ij")
        inside = Path(ring).contains_points(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        for hole in g["coordinates"][1:]: inside &= ~Path(en(hole)).contains_points(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        H[i0:i1, j0:j1] = np.where(inside, np.maximum(H[i0:i1, j0:j1], h), H[i0:i1, j0:j1])
        blds.append(dict(name=p.get("name"), kind=p.get("building"), levels=p.get("building:levels"), h=h, ring=ring, c=ring.mean(0)))
    # stadium from the CAD mesh (stands, pavilions, canopies, upper-deck edges, video boards)
    m = json.load(open(HERE.parent / "KyleField_demo_mesh.json")); u = m["unit_mm"]/1000
    CT = np.zeros((NX, NX), np.float32); CB = np.full((NX, NX), 999.0, np.float32)   # canopies: overhangs (open space below)
    for gi in (4, 5, 6, 7, 8):
        gg = m["groups"][gi]; v = np.array(gg["v"], float).reshape(-1, 3)*u; fc = np.array(gg["f"], int).reshape(-1, 3)
        for t in v[fc]:
            P = cad2en(t[:, 0], t[:, 1]); z = t[:, 2]
            x0, y0 = P.min(0); x1, y1 = P.max(0)
            i0, i1 = np.searchsorted(gx, [x0-CELL, x1+CELL]); j0, j1 = np.searchsorted(gx, [y0-CELL, y1+CELL])
            XX, YY = np.meshgrid(gx[i0:i1], gx[j0:j1], indexing="ij"); q = np.c_[XX.ravel(), YY.ravel()]
            A, B, C = P; v0, v1 = C-A, B-A; w = q-A
            d00, d01, d11 = v0@v0, v0@v1, v1@v1; den = d00*d11-d01*d01
            if abs(den) < 1e-6 and gi == 6: continue      # canopy edges: the top/bottom faces carry the footprint
            if abs(den) < 1e-6:            # vertical face: mark its footprint line with its top height
                for s in np.linspace(0, 1, 40):
                    for a_, b_ in ((A, B), (B, C), (C, A)):
                        pt = a_+(b_-a_)*s; i, j = int((pt[0]+EXT)/CELL), int((pt[1]+EXT)/CELL)
                        if 0 <= i < NX and 0 <= j < NX: H[i, j] = max(H[i, j], z.max())
                continue
            uu = (d11*(w@v0)-d01*(w@v1))/den; vv = (d00*(w@v1)-d01*(w@v0))/den
            ins = (uu >= -0.02) & (vv >= -0.02) & (uu+vv <= 1.02)
            zz = z[0] + uu*(z[2]-z[0]) + vv*(z[1]-z[0])
            if gi == 6:
                a1 = CT[i0:i1, j0:j1].ravel(); a1 = np.where(ins, np.maximum(a1, zz), a1); CT[i0:i1, j0:j1] = a1.reshape(XX.shape)
                a2 = CB[i0:i1, j0:j1].ravel(); a2 = np.where(ins, np.minimum(a2, zz), a2); CB[i0:i1, j0:j1] = a2.reshape(XX.shape)
                continue
            sub = H[i0:i1, j0:j1].ravel(); sub = np.where(ins, np.maximum(sub, zz), sub); H[i0:i1, j0:j1] = sub.reshape(XX.shape)
    return H, blds, gx, CT, CB
def height_at(H, x, y):
    i = np.clip(((x+EXT)/CELL).astype(int), 0, NX-1); j = np.clip(((y+EXT)/CELL).astype(int), 0, NX-1); return H[i, j]
def visible(H, src, T, step=3.0, skip=6.0):
    """src (x,y,z); T (n,3) targets. True where the straight line clears the height map (ignores the first `skip` metres: the mount itself)."""
    out = np.ones(len(T), bool); s = np.asarray(src, float)
    for k in range(0, len(T), 1500):
        t = T[k:k+1500]; D = t - s; L = np.hypot(D[:, 0], D[:, 1]); n = int(L.max()/step)+2
        f = np.linspace(0, 1, n)[None, :]
        px = s[0]+D[:, 0:1]*f; py = s[1]+D[:, 1:2]*f; pz = s[2]+D[:, 2:3]*f
        dist = L[:, None]*f; mask = (dist > skip) & (dist < L[:, None]-2)
        out[k:k+1500] = ~((pz < height_at(H, px, py)) & mask).any(1)
    return out
