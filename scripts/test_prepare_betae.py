import datetime
import hashlib
import pickle
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import prepare_betae as betae


class PrepareTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.target = self.data / betae.DATASET
        self.archive = self.data / "KG_data.zip"
        self.values = defaultdict(set, {("e", ("r",)): {(1, (2,))}})
        self.official = self.root / "fixture.zip"
        self.make_archive(self.official)
        self.expected = hashlib.sha256(self.official.read_bytes()).hexdigest()
        self.fetches = 0

    def make_archive(self, path, unusual_value=None):
        with ZipFile(path, "w") as bundle:
            for name in betae.RAW:
                value = self.values
                if name == "valid-queries.pkl" and unusual_value is not None:
                    value = unusual_value
                contents = pickle.dumps(value, protocol=3) if name.endswith(".pkl") else b"1 2 3\n"
                bundle.writestr(f"{betae.DATASET}/{name}", contents)

    def fetch(self, path):
        self.fetches += 1
        path.write_bytes(self.official.read_bytes())

    def run_setup(self):
        betae.prepare(self.data, self.expected, self.fetch)

    def test_partial_legacy_install_and_corrupt_cache_recover(self):
        self.archive.write_bytes(b"incomplete download")
        self.target.mkdir()
        (self.target / "test-queries.pkl").write_bytes(b"incomplete extraction")
        (self.target / "test-queries.plain.pkl").write_bytes(b"incomplete conversion")
        (self.target / "mine.txt").write_text("keep")
        self.run_setup()
        self.assertEqual(self.fetches, 1)
        self.assertEqual((self.target / "mine.txt").read_text(), "keep")
        for name in betae.PLAIN:
            with (self.target / name).open("rb") as source:
                result = betae.BenchmarkUnpickler(source).load()
            self.assertIs(type(result), dict)
            self.assertEqual(result, self.values)

        # A complete install works offline, even if its download was removed.
        saved = self.archive.read_bytes()
        self.archive.unlink()
        self.run_setup()
        self.assertEqual(self.fetches, 1)
        # A missing output must still be repaired from the verified cache.
        self.archive.write_bytes(saved)
        (self.target / "id2rel.pkl").unlink()
        self.run_setup()
        self.assertTrue((self.target / "id2rel.pkl").is_file())
        self.assertEqual(self.fetches, 1)

    def test_bad_download_leaves_no_cached_archive_or_receipt(self):
        def bad_fetch(path):
            path.write_bytes(b"incomplete download")

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            betae.prepare(self.data, self.expected, bad_fetch)
        self.assertFalse(self.archive.exists())
        self.assertFalse((self.target / betae.RECEIPT).exists())
        self.run_setup()

    def test_unexpected_data_class_leaves_existing_files_alone(self):
        self.make_archive(self.archive, datetime.date(2020, 1, 1))
        self.target.mkdir()
        sentinel = self.target / "test-queries.plain.pkl"
        sentinel.write_bytes(b"old contents")
        with self.assertRaisesRegex(pickle.UnpicklingError, "datetime.date"):
            betae.prepare(self.data, betae.digest(self.archive), self.fetch)
        self.assertEqual(sentinel.read_bytes(), b"old contents")
        self.assertFalse((self.target / betae.RECEIPT).exists())
        self.assertEqual(self.fetches, 0)
        self.run_setup()
        self.assertNotEqual(sentinel.read_bytes(), b"old contents")


if __name__ == "__main__":
    unittest.main()
