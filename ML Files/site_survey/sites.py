import numpy as np, pickle, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent          # this folder (site_survey/)
sys.path.insert(0, str(HERE)); import world as W
H = np.load(HERE / "H.npy"); B = pickle.load(open(HERE / "blds.pkl", "rb"))
br = np.radians(np.arange(0, 360, 2.0)); rr = np.arange(160, 1301, 25.0)
BR, RR = np.meshgrid(br, rr, indexing="ij"); TX = RR.ravel()*np.sin(BR.ravel()); TY = RR.ravel()*np.cos(BR.ravel())
keep = W.height_at(H, TX, TY) < 1                       # targets over open ground only (not inside a building)
TX, TY, TR = TX[keep], TY[keep], RR.ravel()[keep]; WGT = TR/TR.sum()   # area weighting
ALTS = (15, 30, 60)
sites = []
def add(name, kind, x, y, z): sites.append(dict(name=name, kind=kind, x=float(x), y=float(y), z=float(z)))
ROOFS = ["University Center Garage", "Gene Stallings Blvd Garage", "West Campus Garage", "Southside Parking Garage", "Central Campus Garage",
         "Rudder Tower", "Texas A&M Hotel and Conference Center", "Memorial Student Center", "Koldus Building", "Reed Arena",
         "Student Recreation Center", "McFerrin Athletic Center-Indoor Football", "Bright Football Complex", "Northside Garage", "Polo Road Garage"]
for nm in ROOFS:
    for b in B:
        if b["name"] == nm:
            c = b["ring"].mean(0); add(nm, "roof", c[0], c[1], b["h"] + 3.0); break
for b in B:   # unnamed garages
    if b["kind"] == "parking" and not b["name"] and np.hypot(*b["ring"].mean(0)) < 900:
        c = b["ring"].mean(0); add("Unnamed garage %.0f m brg %.0f" % (np.hypot(*c), np.degrees(np.arctan2(*c)) % 360), "roof", c[0], c[1], b["h"] + 3)
for side, yy in (("west", 117), ("east", -117)):
    p = W.cad2en(np.array([0.0]), np.array([float(yy)]))[0]; add(f"Stadium canopy top ({side})", "stadium", p[0], p[1], 63)
for nm, (cx, cy) in {"N-W corner roof": (87.9, 100), "N-E corner roof": (87.9, -99.9), "S-W corner roof": (-88.8, 100), "S-E corner roof": (-88.8, -99.9)}.items():
    p = W.cad2en(np.array([cx]), np.array([cy]))[0]; add(f"Stadium {nm}", "stadium", p[0], p[1], 41.5)
g = np.arange(-700, 701, 40.0)
for x in g:
    for y in g:
        r = np.hypot(x, y)
        if 170 < r < 700:
            xx, yy = np.meshgrid(x + np.arange(-10, 11, 2.0), y + np.arange(-10, 11, 2.0))
            if W.height_at(H, xx.ravel(), yy.ravel()).max() < 0.5: add("Trailer mast %.0f m brg %.0f" % (r, np.degrees(np.arctan2(x, y)) % 360), "trailer", x, y, 12.0)
print(len(sites), "candidate sites;", sum(s["kind"] == "trailer" for s in sites), "trailer spots;", len(TX), "targets")
VIS = {a: np.zeros((len(sites), len(TX)), bool) for a in ALTS}
for i, s in enumerate(sites):
    for a in ALTS:
        T = np.c_[TX, TY, np.full(len(TX), float(a))]
        VIS[a][i] = W.visible(H, (s["x"], s["y"], s["z"]), T, skip=8.0 if s["kind"] != "trailer" else 3.0)
    if i % 50 == 0: print(i, file=sys.stderr)
pickle.dump(dict(sites=sites, VIS=VIS, TX=TX, TY=TY, TR=TR, WGT=WGT), open(HERE / "vis.pkl", "wb"))
