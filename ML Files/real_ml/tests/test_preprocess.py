"""Small validation inputs exercise failure handling; no synthetic training data."""
import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import normalize_label, session_identity, summarize, verify_labels, deduplicate_tracks, annotation_map, ambiguous_label, curated_truth


class PreprocessTests(unittest.TestCase):
    def test_person_dog_is_ambiguous_despite_official_normalization(self):
        self.assertTrue(ambiguous_label("HUMAN/DOG"))
        self.assertFalse(ambiguous_label("HUMAN"))

    def test_curated_metadata_joins_sensor_ids_then_system_id(self):
        # Actual ID relationships in NRC's truck extract, reduced to three validation rows.
        frame = pd.DataFrame({"id": [10, 17, 142], "SystemTrackID": [1259, 1259, 1259]})
        expected = {"10": "VEHICLE", "17": "VEHICLE", "142": "VEHICLE"}
        filename = "Echodyne_Log_20210226_121014_TRUCK_1259.csv"
        self.assertEqual(curated_truth(frame, expected, filename), {"1259": "VEHICLE"})
        frame.loc[0, "SystemTrackID"] = 999
        with self.assertRaises(ValueError):
            curated_truth(frame, expected, filename)
        with self.assertRaises(ValueError):
            curated_truth(frame, {"1259": "VEHICLE"}, filename)

    def test_blank_annotation_header(self):
        self.assertEqual(annotation_map(pd.DataFrame({"SystemTrackID": [29], "Unnamed: 1": ["UAV"]})), {"29": "UAV"})

    def test_duplicate_annotation_is_one_track(self):
        self.assertEqual(annotation_map(pd.DataFrame({"SystemTrackID": [13, 13], "Classification": ["HUMAN", "HUMAN"]})), {"13": "HUMAN"})

    def test_normalize_official_spellings(self):
        for raw, expected in [(" Burd ", "BIRD"), ("Human/dog", "HUMAN"),
                              ("Truck", "VEHICLE"), (None, "UNKNOWN"),
                              ("mix of birds, people,car", "MIXED")]:
            self.assertEqual(normalize_label(raw), expected)
        with self.assertRaises(ValueError):
            normalize_label("CLASSIF_UAV")

    def test_session_alias_and_radar_grouping(self):
        a = session_identity("tracks_2019-10-11T135549-R1.xlsx")
        b = session_identity("tracks_2019-10-11T135549-R2.xlsx")
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        self.assertEqual(session_identity("Echodyne_Log_20210318_102509.xlsx"),
                         session_identity("Echodyne_Log_20210318_102509_UAV_1.csv"))

    def test_summary_invalid_and_missing_are_not_zero(self):
        summary, invalid = summarize(pd.DataFrame({"range": [10, 20, -1, np.inf],
                                                  "rcs": [-30, np.nan, -10, np.nan],
                                                  "probUav": [1, 1, 1, 1]}))
        self.assertEqual(summary["range_mean"], 15)
        self.assertEqual(summary["range_std"], 5)
        self.assertEqual(summary["range_range"], 10)
        self.assertEqual(summary["range_missing_rate"], .5)
        self.assertEqual(summary["range_n_valid"], 2)
        self.assertEqual(summary["rcs_mean"], -20)
        self.assertTrue(np.isnan(summary["posX_mean"]))
        self.assertEqual(invalid["range"], 2)
        self.assertFalse(any("prob" in c for c in summary))

    def test_metadata_disagreement_rejected(self):
        self.assertEqual(verify_labels({"6": "Bird"}, {"6": "BIRD"}), {"6": "BIRD"})
        with self.assertRaises(ValueError):
            verify_labels({"6": "Bird"}, {"6": "UAV"})
        with self.assertRaises(ValueError):
            verify_labels({"6": "Bird"}, {"7": "BIRD"})

    def test_duplicates_prefer_workbook_and_reject_conflicts(self):
        rows = [dict(track_uid="s:R1:1", label="UAV", source_file="a.csv", source_format="csv"),
                dict(track_uid="s:R1:1", label="UAV", source_file="a.xlsx", source_format="xlsx")]
        kept, audit = deduplicate_tracks(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source_file"], "a.xlsx")
        self.assertEqual(len(audit), 1)
        rows[0]["label"] = "BIRD"
        kept, audit = deduplicate_tracks(rows)
        self.assertEqual(kept, [])
        self.assertEqual(len(audit), 2)


if __name__ == "__main__":
    unittest.main()
