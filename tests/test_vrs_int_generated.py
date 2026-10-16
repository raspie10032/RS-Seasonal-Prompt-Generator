"""Metadata direction, identity preservation, resumability and v1/v2 isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, PngImagePlugin

from vrs_int.generation_metadata import extract, make_record, prompt_tags
from vrs_int.generated import build, scan
from vrs_int.manifest import read_pairs
from vrs_int.store import Store


class MetadataTests(unittest.TestCase):
    def test_oom_reduces_batch_without_losing_or_reordering_images(self):
        from vrs_int.generated import bounded_score
        class OutOfMemoryError(RuntimeError): pass
        calls, maximum = [], [8]
        def score(items):
            calls.append(len(items))
            if len(items) > 2: raise OutOfMemoryError()
            return [x * 2 for x in items]
        self.assertEqual(bounded_score(score, list(range(9)), maximum, release=lambda: None), list(range(0,18,2)))
        self.assertEqual(maximum, [2])
        self.assertEqual(calls, [8,4,2,2,2,2,1])

    def test_pixai_preserves_style_and_copyright_but_not_as_verified_ratings(self):
        from vrs_int.generated_tagger import prediction_record
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "a.png"
            Image.new("RGB", (8,8), "red").save(image)
            labels = [{"name": name, "category": kind} for name,kind in
                      [("sample_style","style"),("sample_series","copyright"),("sample_hero","character"),
                       ("red_hair","general"),("rating:e","rating"),("highres","meta")]]
            provenance = {"family":"pixai", "thresholds":dict.fromkeys([x['category'] for x in labels], .2)}
            row = prediction_record(image,"a"*64,labels,[.9]*6,provenance)
            self.assertEqual(row['tags_artist'], ['sample style'])
            self.assertEqual(row['tags_copyright'], ['sample series'])
            self.assertEqual(row['tags_character'], ['sample hero'])
            self.assertNotIn('rating:e', row['caption'])
            self.assertNotIn('highres', row['caption'])
            self.assertEqual(row['rating'], 'u')
            self.assertEqual(row['caption_source'], 'pixai-visual-tag-prediction')
            self.assertEqual(len(row['tagger_prediction']['predictions']),6)

    def test_novelai_characters_negative_and_core_names(self):
        parsed = extract({"Comment": json.dumps({"prompt": "fallback", "uc": "rain, blur",
            "v4_prompt": {"caption": {"base_caption": "artist:sample_artist, 1.2::blue_sky::",
                "char_captions": [{"char_caption": "sample_hero_(series), red_hair", "centers": [{"x": .5, "y": .5}]}]}},
            "api_token": "SHOULD_NOT_BE_COPIED"}), "unrelated_secret": "DO_NOT_COPY"})
        row = make_record(parsed, "a.png", "a" * 64, {"sample hero (series)": ["character"]})
        self.assertIn("sample artist", row["tags_artist"])
        self.assertIn("sample hero (series)", row["tags_character"])
        self.assertNotIn("rain", row["tags"])
        self.assertNotIn("SHOULD_NOT_BE_COPIED", json.dumps(row))
        self.assertIn("rain, blur", parsed["negative"])
        self.assertEqual(row["rating"], "u")
        self.assertIn("Character 1:", row["caption"])

    def test_comfy_uses_connected_positive_and_realized_wildcard(self):
        graph = {
            "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "test"}},
            "2": {"class_type": "KSampler", "inputs": {"positive": ["3", 0], "negative": ["4", 0]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "rain, bad hands"}},
            "5": {"class_type": "ImpactWildcardProcessor", "inputs": {"populated_text": "blue sky, red dress", "wildcard_text": "{winter|summer}"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "unused character"}},
        }
        result = extract({"prompt": json.dumps(graph)}, "test_00001.png")
        self.assertEqual(result["positive"], ["blue sky, red dress"])
        self.assertEqual(result["negative"], ["rain, bad hands"])
        self.assertNotIn("winter", json.dumps(result))
        graph["7"] = {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "other"}}
        self.assertIsNone(extract({"prompt": json.dumps(graph)}, "renamed.png"))

    def test_weights_do_not_destroy_qualified_names(self):
        tags, kinds, rejected = prompt_tags(["{{artist:sample_artist}}, (red_dress, blue_sky:1.2), hero_(series), {winter|summer}, best_quality"])
        self.assertIn("hero (series)", tags)
        self.assertIn("red dress", tags)
        self.assertEqual(kinds["sample artist"], ["artist"])
        self.assertNotIn("winter|summer", tags)
        self.assertTrue(rejected)

    def test_nonpositive_groups_are_preserved_but_never_positive_tags(self):
        prompt = ("artist:kept, 1girl, -2::hat, artist:suppressed, character:hidden::, "
                  "0::copyright:zero_series::, .5::blue_sky::, hero_(series), "
                  "(red_dress, rain:-1), green_eyes, -1::snow")
        generation = {"positive": [prompt], "characters": [], "negative": ["blur"]}
        row = make_record(generation, "a.png", "a" * 64)
        self.assertEqual(row["tags"], ["1girl", "blue sky", "green eyes", "hero (series)", "kept"])
        self.assertEqual(row["tags_artist"], ["kept"])
        self.assertEqual(row["caption"], prompt)
        self.assertEqual(row["generation"], generation)
        saved = json.dumps(row["suppressed_prompt_fragments"])
        for value in ("hat", "artist:suppressed", "character:hidden", "zero_series", "red_dress", "snow"):
            self.assertIn(value, saved)

    def test_negative_weight_does_not_remove_separate_positive_occurrence(self):
        tags, kinds, _ = prompt_tags(["-1::artist:repeat, rain::, artist:repeat, blue_sky"])
        self.assertEqual(tags, ["blue sky", "repeat"])
        self.assertEqual(kinds["repeat"], ["artist"])

    def test_old_generated_recommendation_view_preserves_original_and_vector(self):
        from copy import deepcopy
        from vrs_int.generation_metadata import recommendation_view
        generation = {"positive": ["artist:kept, -1::artist:suppressed, hat::, hero_(series)"],
                      "characters": [], "negative": []}
        source = make_record(generation, "a.png", "a" * 64)
        source.update(metadata_parser="generation-metadata-v2", tags=["kept", "suppressed", "hat", "hero (series)"],
                      tags_artist=["kept", "suppressed"])
        entry = {"document": {"tags": source['tags']}, "vector": {"sentinel": "unchanged"},
                 "categories": {"kept": ["artist"], "suppressed": ["artist"], "hero (series)": ["character"]},
                 "audit": [{"source_record": source}]}
        before = deepcopy(entry)
        result = recommendation_view(entry)
        self.assertEqual(result["document"]["tags"], ["hero (series)", "kept"])
        self.assertEqual(result["categories"]["hero (series)"], ["character"])
        self.assertEqual(entry, before)
        self.assertIs(result["audit"], entry["audit"])
        self.assertIs(result["vector"], entry["vector"])
        # Native pixel provenance is supplemental, not a second prompt source.
        entry["audit"].append({"native_pixel_sha256": "b" * 64})
        native_result = recommendation_view(entry)
        self.assertEqual(native_result["document"]["tags"], result["document"]["tags"])
        self.assertIs(native_result["audit"], entry["audit"])
        self.assertIs(native_result["vector"], entry["vector"])
        entry["audit"].append({"source_record": source})
        self.assertIs(recommendation_view(entry), entry)  # Ambiguous source history.
        entry["audit"].pop()
        source["caption_source"] = "pixai-visual-tag-prediction"
        self.assertIs(recommendation_view(entry), entry)

    def test_recommendation_view_retains_legacy_json_audit(self):
        from vrs_int.generation_metadata import recommendation_view
        entry = {"document": {"tags": ["example artist"]},
                 "vector": {"sentinel": "unchanged"},
                 "categories": {"example artist": ["artist"]},
                 "audit": [{"source_record": '{"id":1,"tags_artist":["example_artist"]}'}]}
        self.assertIs(recommendation_view(entry), entry)
        self.assertIsInstance(entry["audit"][0]["source_record"], str)

    def test_renamed_comfy_outputs_share_only_common_prompts(self):
        graph = {
            "1": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "base"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "final"}},
            "3": {"class_type": "KSampler", "inputs": {"positive": "blue sky", "negative": "rain"}},
            "4": {"class_type": "FaceDetailer", "inputs": {"image": ["3", 0], "positive": "red eyes", "negative": "blur"}},
        }
        result = extract({"prompt": json.dumps(graph)}, "renamed.png")
        self.assertEqual(result["positive"], ["blue sky"])
        self.assertNotIn("red eyes", make_record(result, "renamed.png", "a" * 64)["tags"])
        self.assertEqual(result["output_selection"], "common-prompts-across-outputs")

    def test_wd_fallback_preserves_prediction_origin(self):
        from vrs_int.generated_tagger import fallback
        from vrs_int.generated import expand
        import sqlite3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); images = root / "images"; images.mkdir()
            Image.new("RGB", (8, 8), "red").save(images / "a.png")
            catalog = root / "catalog.sqlite3"
            scan([images], catalog)
            class FakeWD:
                provenance = {"test": "not-a-real-model"}
                labels = [{"name": "red_hair", "category": "0"},
                          {"name": "sample_hero", "category": "4"},
                          {"name": "unsafe", "category": "9"}]
                def __init__(self, path, workers): pass
                def score(self, paths): return [[.8, .6, .99] for _ in paths]
            output = root / "wd.sqlite3"
            result = fallback(catalog, output, root, tagger_factory=FakeWD)
            self.assertEqual(result["unique_records"], 1)
            self.assertEqual(fallback(catalog, output, root, tagger_factory=FakeWD)["processed_this_run"], 0)
            with sqlite3.connect(output) as db:
                row = expand(db.execute("SELECT record FROM records").fetchone()[0])
            self.assertEqual(row["tags"], ["red hair"])
            self.assertEqual(row["tags_artist"], [])
            self.assertEqual(row["tags_character"], [])
            self.assertEqual(row["rating"], "u")
            self.assertEqual(row["caption_source"], "wd-visual-tag-prediction")

    def test_a1111_negative_is_not_caption(self):
        parsed = extract({"parameters": "blue sky, 1girl\nNegative prompt: rain, blur\nSteps: 20, Seed: 42"})
        self.assertEqual(parsed["positive"], ["blue sky, 1girl"])
        self.assertEqual(parsed["negative"], ["rain, blur"])
        self.assertIsNone(extract({"Description": "unverified arbitrary note"}))


class GeneratedPipelineTests(unittest.TestCase):
    def test_scan_resume_duplicate_build_resume_and_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); images = root / "images"; images.mkdir()
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Comment", json.dumps({"prompt": "artist:sample_artist, blue sky", "uc": "rain"}))
            Image.new("RGB", (16, 16), "blue").save(images / "a.png", pnginfo=metadata)
            (images / "copy.png").write_bytes((images / "a.png").read_bytes())
            Image.new("RGB", (16, 16), "red").save(images / "no_metadata.png")
            catalog = root / "catalog.sqlite3"
            result = scan([images], catalog)
            self.assertEqual(result["unique_records"], 1)
            self.assertEqual(result["files"], {"duplicate-bytes": 1, "included": 1, "no-generation-metadata": 1})
            self.assertEqual(scan([images], catalog)["unique_records"], 1)
            self.assertEqual(len(list(read_pairs(catalog.with_suffix(".jsonl"), Path(root.anchor)))), 1)
            encoder = root / "encoder"; encoder.mkdir()
            (encoder / "provenance.json").write_text("{}")
            calls = []
            class FakeEncoder:
                def __init__(self, directory): pass
                def encode(self, image):
                    calls.append(image)
                    return np.ones(768, dtype=np.float32) / np.sqrt(768)
            output = root / "store"; cache = root / "checkpoint.sqlite3"
            first = build(catalog, encoder, output, cache, delay=0, encoder_factory=FakeEncoder)
            second = build(catalog, encoder, output, cache, delay=0, encoder_factory=FakeEncoder)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first["snapshot"], second["snapshot"])
            store = Store(output)
            self.assertEqual(store.manifest(first["snapshot"])["format"], "vrs-int/2")
            entry = next(store.entries(first["snapshot"]))
            self.assertTrue(entry["key"].startswith("generated:"))
            self.assertIn("sample artist", entry["document"]["tags"])
            self.assertNotIn("rain", entry["document"]["tags"])
            package = root / "package"
            store.export(first["snapshot"], package)
            recipient = Store(root / "recipient")
            self.assertEqual(recipient.install(package), first["snapshot"])
            self.assertEqual(recipient.verify(first["snapshot"])["vectors"], 1)
            from vrs_int.generated_pool import merge_stores
            part = dict(store=str(output), snapshot=first['snapshot'])
            merged = merge_stores([part], root/'merged')
            self.assertEqual(merged['vectors'],1)
            self.assertEqual(Store(root/'merged').current(),merged['snapshot'])
            self.assertEqual(next(Store(root/'merged').entries(merged['snapshot']))['audit'],entry['audit'])
            with self.assertRaisesRegex(ValueError,'Duplicate observation'):
                merge_stores([part,part],root/'conflicting')
            self.assertIsNone(Store(root/'conflicting').current())

            # A valid content-addressed store alone cannot prove that captions
            # still match the ingestion catalog. Audit that boundary separately.
            from vrs_int.generated_audit import audit_run
            from vrs_int.generated import compact, expand
            import sqlite3
            run = root/'audit-run'; run.mkdir()
            (run/'report.json').write_text(json.dumps(dict(merged,state='complete')))
            db = sqlite3.connect(run/'pixai.sqlite3')
            db.execute('CREATE TABLE records(sha TEXT PRIMARY KEY, record BLOB)')
            db.commit(); db.close()
            self.assertEqual(audit_run(catalog,run)['source_records_compared'],1)
            db = sqlite3.connect(catalog)
            sha, payload = db.execute('SELECT sha,record FROM records').fetchone()
            altered = expand(payload); altered['caption'] = 'a different original caption'
            db.execute('UPDATE records SET record=? WHERE sha=?',(compact(altered),sha))
            db.commit(); db.close()
            with self.assertRaisesRegex(ValueError,'Original caption'):
                audit_run(catalog,run)


if __name__ == "__main__":
    unittest.main()
