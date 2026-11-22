"""Optional integration checks against the downloaded official snapshot and outputs."""
from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import BASE


@unittest.skipUnless((BASE / "data/processed/tracks.csv").exists(), "Run the real pipeline first")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracks = pd.read_csv(BASE / "data/processed/tracks.csv")

    def test_verified_snapshot_counts(self):
        self.assertEqual(self.tracks.label.value_counts().to_dict(),
                         {"BIRD": 492, "UAV": 146, "HUMAN": 63, "VEHICLE": 12})
        self.assertEqual(int(self.tracks.n_observations.sum()), 367112)
        self.assertTrue(self.tracks.track_uid.is_unique)

    def test_curated_duplicates_and_ambiguity_excluded(self):
        rows = self.tracks[self.tracks.track_uid == "NRC:20210318T102509:R1:1"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0].source_format, "xlsx")
        self.assertNotIn("NRC:20201218T145654:R1:195", set(self.tracks.track_uid))

    def test_every_column_documented_and_no_radar_predictions(self):
        dictionary = pd.read_csv(BASE / "reports/feature_dictionary.csv")
        self.assertEqual(list(dictionary.column), list(self.tracks.columns))
        self.assertEqual(len(self.tracks.columns), 79)
        for forbidden in ["probUav", "probUnknow", "UpdatedTrackType", "TrackExtendedNature", "confidenceLevel"]:
            self.assertFalse(any(forbidden in c for c in self.tracks.columns))

    def test_workbook_track_mean_matches_real_measurements(self):
        path = BASE / "data/raw/data/Demo-0/tracks_2019-10-11T134524-R1.xlsx"
        if not path.exists():
            self.skipTest("Official workbook not downloaded")
        raw = pd.read_excel(path)
        selected = raw[raw.SystemTrackID == 6]
        actual = self.tracks.set_index("track_uid").loc["NRC:20191011T134524:R1:6"]
        self.assertEqual(actual.n_observations, len(selected))
        self.assertAlmostEqual(actual.range_mean, selected["range"].mean(), places=7)
        self.assertAlmostEqual(actual.rcs_missing_rate, selected.rcs.isna().mean(), places=10)


if __name__ == "__main__":
    unittest.main()
