"""Publication checks use isolated Git indexes, with no personal data or model."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_code_only.py"


class PublicationTests(unittest.TestCase):
    def test_checks_staged_contents_and_rejects_private_or_replaced_fixture(self):
        env = dict(os.environ)
        for key in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
            env.pop(key, None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)

            def check():
                return subprocess.run([sys.executable, str(CHECKER)], cwd=root, env=env,
                                      capture_output=True, text=True)

            git("init", "-q")
            (root / "app.py").write_text('print("hello")\n', encoding="utf-8")
            git("add", "app.py")
            private = root / "memory" / "objects" / "private.z"
            private.parent.mkdir(parents=True)
            private.write_bytes(b"private content placeholder")
            self.assertEqual(check().returncode, 0)  # untracked private files stay local
            git("add", "memory")
            private.unlink()  # deleting a working file does not remove staged data
            failed = check()
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("private.z", failed.stderr)
            self.assertNotIn("private content placeholder", failed.stderr)
            git("rm", "--cached", "memory/objects/private.z")
            fixture = root / "examples/vrs-int/batch-01.jsonl"
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes((ROOT / fixture.relative_to(root)).read_bytes())
            git("add", "examples")
            self.assertEqual(check().returncode, 0)
            fixture.write_text('{"rating":"q","tags":["replacement"]}\n', encoding="utf-8")
            git("add", "examples")
            self.assertNotEqual(check().returncode, 0)

    def test_catalogues_cache_vectors_and_sqlite_journals_are_not_code(self):
        from scripts.check_code_only import check_blob
        for path in ("custom/vectors.f32", "custom/catalog.sqlite3-wal", "custom/catalog.db-journal",
                     "rs_vrs_folders/collection/notes.txt", "persistent-v1/manifest.txt",
                     "RS_VRS_FOLDERS/collection/notes.txt", "vrs-int-generated/receipt.txt",
                     "custom/image.avif", "custom/image.TIFF"):
            with self.subTest(path=path):
                self.assertIsNotNone(check_blob(path, "100644", b"fictional placeholder"))


if __name__ == "__main__":
    unittest.main()
