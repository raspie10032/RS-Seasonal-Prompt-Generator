"""Local-only data paths, tested with fictional records and blocked networking."""
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrs_int.folders import FolderCatalog
from vrs_int.privacy import IGNORE, private_directory
from vrs_int.search_index import SearchIndex
from vrs_int.store import Store
from test_vrs_int import contract, entry


class LocalPrivacyTests(unittest.TestCase):
    def test_local_lifecycle_ignores_metadata_even_inside_another_git_repo(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE")}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, env=env)

            git("init", "-q")
            with patch.dict(os.environ, {"RS_VRS_CACHE_DIR": str(root / "custom-cache")}), \
                    patch("vrs_int.search_cache.shutil.disk_usage", return_value=SimpleNamespace(
                        total=100 * 1024**3, free=80 * 1024**3)), \
                    patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), \
                    patch("socket.socket.connect_ex", side_effect=AssertionError("network forbidden")), \
                    patch("socket.getaddrinfo", side_effect=AssertionError("DNS forbidden")):
                store = Store(root / "custom-memory")
                snapshot = store.build([entry()], contract(), "fictional", bucket_count=2)
                store.update(snapshot, [], "unchanged")
                store.activate(snapshot, None)
                store.export(snapshot, root / "custom-backup")
                imported = Store(root / "custom-restored")
                self.assertEqual(imported.install(root / "custom-backup"), snapshot)
                imported.verify(snapshot)
                self.assertEqual(imported.manifest(snapshot)["counts"]["observations"], 1)
                index = SearchIndex(store, snapshot)
                try:
                    self.assertEqual(index.count, 1)
                    self.assertEqual(len(index.documents([0])), 1)
                finally:
                    index.close()
                images = root / "empty-images"
                images.mkdir()
                catalog = FolderCatalog.create(root / "custom-folders", [str(images)], "fictional.safetensors")
                list(catalog.scan_steps())
                self.assertTrue((catalog.directory / "config.json").is_file())
            (root / "app.py").write_text("# public code only\n", encoding="utf-8")
            git("add", ".")
            self.assertEqual(git("ls-files", "-z"), b"app.py\x00")
            # Store reads do not rewrite even the ignore file.
            marker = store.root / ".gitignore"
            marker.unlink()
            store.verify(snapshot)
            self.assertFalse(marker.exists())

    def test_existing_rules_survive_and_later_unignore_cannot_hide_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / ".gitignore"
            marker.write_bytes(b"# user comment\n!receipt.txt")
            private_directory(root)
            first = marker.read_bytes()
            self.assertTrue(first.startswith(b"# user comment\n!receipt.txt"))
            private_directory(root)
            self.assertEqual(marker.read_bytes(), first)
            with marker.open("ab") as stream:
                stream.write(b"!metadata.json\n")
            private_directory(root)
            self.assertTrue(marker.read_bytes().endswith(IGNORE))

    def test_refuses_repo_root_or_linked_ignore_before_writing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "repository root"):
                private_directory(root)
            self.assertFalse((root / ".gitignore").exists())
            folder = root / "data"
            folder.mkdir()
            target = root / "existing.txt"
            target.write_text("keep", encoding="utf-8")
            try:
                (folder / ".gitignore").symlink_to(target)
            except OSError:
                self.skipTest("Symlinks unavailable")
            with self.assertRaisesRegex(ValueError, "regular local file"):
                private_directory(folder)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_protection_failure_precedes_store_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "memory"
            with patch("vrs_int.privacy.os.open", side_effect=PermissionError("cannot protect data")):
                with self.assertRaises(PermissionError):
                    Store(root).build([entry()], contract(), "fictional")
            self.assertFalse((root / "objects").exists())
            self.assertFalse((root / "snapshots").exists())


if __name__ == "__main__":
    unittest.main()
