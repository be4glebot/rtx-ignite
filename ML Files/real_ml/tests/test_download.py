import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_data import git_blob_sha, safe_path


class DownloadTests(unittest.TestCase):
    def test_git_blob_integrity(self):
        self.assertEqual(git_blob_sha(b"test\n"), "9daeafb9864cf43055ae93beb0afd6c7d144bfa4")

    def test_reject_path_escape(self):
        with self.assertRaises(ValueError):
            safe_path(Path.cwd(), "../escape")
        with self.assertRaises(ValueError):
            safe_path(Path.cwd(), "C:/escape")


if __name__ == "__main__":
    unittest.main()
