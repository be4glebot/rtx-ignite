import unittest
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explore import validate_tracks


class ExploreTests(unittest.TestCase):
    def test_duplicate_track_rejected(self):
        with self.assertRaises(ValueError):
            validate_tracks(pd.DataFrame({"track_uid": ["a", "a"], "label": ["UAV", "UAV"]}))

    def test_empty_tracks_rejected(self):
        with self.assertRaises(ValueError):
            validate_tracks(pd.DataFrame({"track_uid": [], "label": []}))


if __name__ == "__main__":
    unittest.main()
