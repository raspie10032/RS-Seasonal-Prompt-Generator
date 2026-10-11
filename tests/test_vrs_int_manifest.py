"""Exercise the published pair examples and their model-free preparation path."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vrs_int.build import build_personal
from vrs_int.manifest import check_manifest
from vrs_int.store import Store


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "vrs-int"
spec = importlib.util.spec_from_file_location("vrs_int_example_manifest", EXAMPLES / "make_manifest.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ExampleEncoder:
    """Storage integration only; not evidence of real image encoding."""
    def __init__(self, directory):
        pass

    def encode(self, image):
        return [1.] + [0.] * 767


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        # Metadata/file-hash checks do not decode pixels.
        for name in ("1001.png", "1002.jpg", "1003.webp"):
            (self.images / name).write_bytes(("test image placeholder: " + name).encode())
        self.rows = [json.loads(line) for line in (EXAMPLES / "batch-01.jsonl").read_text(encoding="utf-8").splitlines()]
        self.manifest = self.root / "input.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows):
        self.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return self.manifest

    def test_templates_build_update_preserve_names_caption_and_previous_version(self):
        encoder = self.root / "encoder"
        encoder.mkdir()
        (encoder / "provenance.json").write_text("{}")
        store = Store(self.root / "memory")
        snapshots = []
        for batch in ("batch-01.jsonl", "batch-02.jsonl"):
            report = build_personal(EXAMPLES / batch, self.images, encoder, store.root,
                                    base=snapshots[-1] if snapshots else None,
                                    encoder_factory=ExampleEncoder)
            snapshots.append(report["snapshot"])
        self.assertEqual(store.verify(snapshots[0])["observations"], 2)
        self.assertEqual(store.verify(snapshots[1])["observations"], 3)
        old = store.get(snapshots[0], "danbooru:1001")
        new = store.get(snapshots[1], "danbooru:1001")
        self.assertNotIn("blue sky", old["document"]["tags"])
        self.assertIn("blue sky", new["document"]["tags"])
        for category, name in (("artist", "example artist a"), ("character", "example character a"),
                               ("copyright", "example series a")):
            self.assertEqual(new["categories"][name], [category])
        self.assertEqual(old["audit"][0]["source_record"]["caption"], self.rows[0]["caption"])
        self.assertNotIn(self.rows[0]["caption"], old["document"]["tags"])
        self.assertIn("signature", old["audit"][0]["removed_tags"])
        self.assertEqual(store.get(snapshots[0], "danbooru:1002"), store.get(snapshots[1], "danbooru:1002"))

    def test_example_sidecars_match_jsonl_and_publish_once(self):
        for source in (EXAMPLES / "pairs").glob("*.json"):
            (self.images / source.name).write_bytes(source.read_bytes())
        output = self.root / "prepared.jsonl"
        report = helper.make_manifest(self.images, output)
        self.assertEqual(report["records"], 2)
        self.assertEqual([json.loads(s) for s in output.read_text(encoding="utf-8").splitlines()], self.rows)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            helper.make_manifest(self.images, output)
        self.assertEqual(output.read_bytes(), before)
        cli_output = self.root / "prepared-stdlib.jsonl"
        run = subprocess.run([sys.executable, "-S", str(EXAMPLES / "make_manifest.py"),
                              "--pairs", str(self.images), "--output", str(cli_output)],
                             cwd=self.root, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(cli_output.read_bytes(), before)

    def test_bad_sidecar_publishes_nothing(self):
        sidecar = self.images / "missing.png.json"
        sidecar.write_text(json.dumps(self.rows[0]))
        output = self.root / "invalid.jsonl"
        with self.assertRaises(ValueError):
            helper.make_manifest(self.images, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".vrs-int-pairs-*")))

    def test_nested_unicode_paths_and_all_ratings(self):
        nested = self.images / "취향별"
        nested.mkdir()
        (nested / "그림.png").write_bytes(b"file bytes")
        rows = [dict(self.rows[0], id=2000 + i, rating=r, image="취향별/그림.png")
                for i, r in enumerate("gsqe")]
        report = check_manifest(self.write_rows(rows), self.images)
        self.assertEqual(report["ratings"], dict.fromkeys("gsqe", 1))
        self.assertEqual(report["typed_unique_tags"]["artist"], 1)

    def test_errors_include_line_and_reject_caption_only_or_ambiguous_tags(self):
        original = self.rows[0]
        variants = [None, dict(original, id=True), dict(original, rating="x"),
                    dict(original, image="missing.png"), dict(original, image="../outside.png"),
                    dict(original, image="C:/images/1001.png"), dict(original, image="C:1001.png"),
                    dict(original, tags_general="blue_hair white_dress"),
                    dict(original, tags_general=[""]), dict(original, tags_artist=["___"]),
                    dict(original, caption=["not a string"]),
                    dict(original, image_sha256="0" * 64), dict(original, split="training"),
                    {"id": 1001, "image": "1001.png", "rating": "g", "caption": "Only prose"}]
        for row in variants:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "Line 1:"):
                check_manifest(self.write_rows([row]), self.images)
        with self.assertRaisesRegex(ValueError, "Line 2: Duplicate"):
            check_manifest(self.write_rows([original, original]), self.images)

    def test_symlink_outside_root_is_rejected(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside image")
        try:
            (self.images / "link.png").symlink_to(outside)
        except OSError:
            self.skipTest("symlinks are unavailable for this user")
        with self.assertRaisesRegex(ValueError, "escapes"):
            check_manifest(self.write_rows([dict(self.rows[0], image="link.png")]), self.images)

    def test_standard_library_only_cli_with_site_packages_disabled(self):
        self.write_rows(self.rows)
        run = subprocess.run([sys.executable, "-S", "-m", "vrs_int", "check-input",
                              "--manifest", str(self.manifest), "--images", str(self.images)],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["records"], 2)
        self.assertFalse((self.root / "memory").exists())
        bad = copy.deepcopy(self.rows)
        bad[1]["image"] = "missing.png"
        self.write_rows(bad)
        run = subprocess.run([sys.executable, "-S", "-m", "vrs_int", "check-input",
                              "--manifest", str(self.manifest), "--images", str(self.images)],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Line 2:", run.stderr)


if __name__ == "__main__":
    unittest.main()
