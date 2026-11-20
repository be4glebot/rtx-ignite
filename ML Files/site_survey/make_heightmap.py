"""Step 1 of the site survey: build the 2 m height map (H.npy) and building list (blds.pkl) from the OSM export
(kf.geojson), origin.json and the stadium mesh (../KyleField_demo_mesh.json)."""
import pickle, numpy as np
from pathlib import Path
import world as W
HERE = Path(__file__).resolve().parent
H, blds, gx, CT, CB = W.load()
np.save(HERE / "H.npy", H); pickle.dump(blds, open(HERE / "blds.pkl", "wb"))
print("wrote H.npy", H.shape, "and blds.pkl with", len(blds), "buildings")
