"""Real line of sight around Kyle Field.

H_sim.npy: 2 m height map in the simulator's frame (x across the field, +x = west; y along the field, +y = north; origin = midfield),
built from our FreeCAD stadium model plus OpenStreetMap building footprints within ~1.7 km (heights from floor counts where
OpenStreetMap has them, otherwise estimates). Flat ground; trees are not included. Outside the map the ground is open.
"""
import numpy as np, os, json
_D = os.path.dirname(os.path.abspath(__file__))
H = np.load(os.path.join(_D, "H_sim.npy")).astype(np.float32)
_C = np.load(os.path.join(_D, "canopy_sim.npy")).astype(np.float32)
CB, CT = _C[0], _C[1]          # the east and west canopies: solid only between their bottom and top (open space underneath)
EXT, CELL = 1700.0, 2.0
N = H.shape[0]
HMAX = float(max(H.max(), CT.max()))
GARAGES = json.load(open(os.path.join(_D, "garages.json")))


def _cell(x, y):
    i = ((x + EXT) / CELL).astype(np.int64); j = ((y + EXT) / CELL).astype(np.int64)
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    return np.where(ok, i, 0), np.where(ok, j, 0), ok


def height_at(x, y):
    i = ((x + EXT) / CELL).astype(np.int64); j = ((y + EXT) / CELL).astype(np.int64)
    ok = (i >= 0) & (i < N) & (j >= 0) & (j < N)
    out = np.zeros(np.shape(x), np.float32)
    out[ok] = H[i[ok], j[ok]]
    return out


def los(src, X, Y, Z, step=4.0, skip=8.0):
    """True where the straight line from src (x, y, z) to each point (X, Y, Z) clears every structure.
    NaN points return False. Only the part of each line inside the map and below the tallest structure is checked."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.nan_to_num(np.asarray(Z, float), nan=-1.0)
    # an object can't be inside a structure: over the stands or a roof, its height counts from that surface (a phone in the
    # stands, a drone skimming a roof). The flight paths are generated without the buildings, so this keeps them physical.
    Z = np.maximum(Z, height_at(np.nan_to_num(X, nan=-1e9), np.nan_to_num(Y, nan=-1e9)) + 2.0)
    out = np.isfinite(X) & np.isfinite(Y)
    sx, sy, sz = map(float, src)
    need = out & ~((Z > HMAX) & (sz > HMAX))
    idx = np.where(need)[0]
    for k in range(0, len(idx), 400):
        ii = idx[k:k + 400]
        dx = X[ii] - sx; dy = Y[ii] - sy; dz = Z[ii] - sz
        L = np.hypot(dx, dy)
        Lc = np.minimum(L, 2600.0)                       # beyond ~2.6 km from any sensor the line has left the map
        n = int(Lc.max() / step) + 2
        f = (np.arange(n)[None, :] * step) / np.maximum(L[:, None], 1e-6)
        valid = (f * L[:, None] > skip) & (f < 1.0 - 2.0 / np.maximum(L[:, None], 1e-6))
        px = sx + dx[:, None] * f; py = sy + dy[:, None] * f; pz = sz + dz[:, None] * f
        valid &= pz < HMAX
        ci, cj, ok = _cell(px, py)
        hb = np.where(ok, H[ci, cj], 0.0); cb = np.where(ok, CB[ci, cj], 999.0); ct = np.where(ok, CT[ci, cj], 0.0)
        blocked = (valid & ((pz < hb) | ((pz >= cb) & (pz <= ct)))).any(1)
        out[ii] = ~blocked
    return out
