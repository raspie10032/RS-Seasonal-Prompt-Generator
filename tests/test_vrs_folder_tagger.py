"""Predictions remain local, distinct, resumable, and subordinate to generation."""
import asyncio
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_vrs_folders as fixtures
from vrs_int.folders import FolderCatalog
from vrs_int.generated_tagger import prediction_record
from vrs_int.generation_metadata import extract
from vrs_int.import_snapshot import file_digest
from vrs_int.store import Store
from vrs_save_node import VRSSaveImage


class FolderTaggerTests(unittest.TestCase):
    setUp = fixtures.FolderTests.setUp
    image = fixtures.FolderTests.image
    scan = fixtures.FolderTests.scan
    service = fixtures.FolderTests.service

    def enable(self):
        self.catalog = FolderCatalog.create(self.root / "collections", [self.images],
                                            "gemma.safetensors", tagger="pixai")

    def row(self, path, sha):
        return prediction_record(path, sha, [{"name": "blue_hair", "category": "general"}],
                                 [.9], {"family": "pixai", "thresholds": {"general": .17}})

    def test_metadata_priority_and_duplicate_prediction_checkpoints(self):
        self.enable()
        metadata = self.image()
        plain = self.image("plain.png", prompt=None, color="green")
        shutil.copy2(plain, self.images / "copy.png")
        before = {p: file_digest(p) for p in self.images.iterdir()}
        self.scan()
        self.assertEqual(len(self.catalog.tag_pending(10)), 1)
        service = self.service()
        service.tagger = Mock(predict=Mock(side_effect=self.row))
        service.ticket = "import"
        snapshot = service.execute("import")["snapshot"]
        service.tagger.predict.assert_called_once()
        self.assertEqual(Store(self.catalog.memory).verify(snapshot)["observations"], 2)
        self.assertEqual(self.catalog.status()["tagging"], {"tagged": 1})
        self.assertEqual(before, {p: file_digest(p) for p in self.images.iterdir()})
        self.scan()
        self.assertFalse(self.catalog.has_pending())
        self.assertEqual(FolderCatalog(self.catalog.directory).status()["tagging"], {"tagged": 1})

    def test_enable_existing_collection_rechecks_only_skipped_images(self):
        self.image()
        self.image("plain.png", prompt=None, color="green")
        self.scan()
        service = self.service()
        service.ticket = "first"
        snapshot = service.execute("first")["snapshot"]
        self.assertFalse(self.catalog.tag_pending(10))
        self.catalog.configure(tagger="pixai", resume_on_start=True)
        self.scan()
        self.assertEqual(len(self.catalog.tag_pending(10)), 1)
        self.assertEqual(self.catalog.status()["records"], {"encoded": 1})
        self.assertEqual(self.catalog.snapshot(), snapshot)
        reopened = FolderCatalog(self.catalog.directory)
        self.assertEqual(reopened.config["tagger"], "pixai")
        self.assertTrue(reopened.config["resume_on_start"])
        with self.assertRaises(ValueError):
            reopened.configure(tagger="remote")
        self.assertEqual(FolderCatalog(self.catalog.directory).config, reopened.config)

    def test_generation_arrival_after_tagging_preserves_prediction_without_vision(self):
        self.enable()
        self.image(prompt=None)
        self.scan()
        service = self.service()
        def predict(path, sha):
            service.server.prompt_queue.put((1, "user"))
            return self.row(path, sha)
        service.tagger = Mock(predict=predict)
        service.ticket = "import"
        self.assertEqual(service.execute("import")["state"], "yielded")
        service.encoder.encode_tensor.assert_not_called()
        row = self.catalog.pending(1)[0][2]
        self.assertEqual(row["caption_source"], "pixai-visual-tag-prediction")
        self.assertEqual(row["rating"], "u")
        service.server.prompt_queue.pending.clear()
        service.ticket = "resume"
        service.tagger.predict = Mock(side_effect=AssertionError("Already checkpointed"))
        snapshot = service.execute("resume")["snapshot"]
        self.assertEqual(Store(self.catalog.memory).verify(snapshot)["observations"], 1)

    def test_stop_during_preparation_does_not_lose_work(self):
        from vrs_folder_service import SearchPreparationYield
        self.enable()
        self.image(prompt=None)
        self.scan()
        service = self.service()
        service.ticket = "import"
        def prepare(checkpoint):
            service.stopped.set()
            checkpoint()
        with patch("vrs_int.folder_tagger.resolve_tagger", side_effect=prepare), \
                patch("vrs_int.folder_tagger.FolderTagger", side_effect=AssertionError("No load")):
            self.assertEqual(service.execute("import")["state"], "yielded")
        self.assertIsNone(service.error)
        self.assertEqual(self.catalog.status()["tagging"], {"pending": 1})

    def test_changed_source_and_no_confident_tags_never_become_experience(self):
        self.enable()
        plain = self.image(prompt=None)
        self.scan()
        service = self.service()
        service.tagger = Mock()
        service.ticket = "first"
        plain.write_bytes(b"changed")
        service.execute("first")
        service.tagger.predict.assert_not_called()
        self.assertEqual(self.catalog.status()["tagging"], {"error": 1})
        self.assertIsNone(self.catalog.snapshot())

    def test_invalid_selection_rejected_and_default_metadata_only(self):
        self.image(prompt=None)
        self.scan()
        self.assertFalse(self.catalog.has_pending())
        with self.assertRaisesRegex(ValueError, "tagger"):
            FolderCatalog.create(self.root, [self.images], "gemma", tagger="http://remote")

    def test_cpu_consent_persists_requires_boolean_and_cannot_mix_encoded_memory(self):
        self.assertFalse(self.catalog.config['use_cpu'])
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.catalog.configure(use_cpu='true')
        self.catalog.configure(use_cpu=True)
        self.assertTrue(FolderCatalog(self.catalog.directory).config['use_cpu'])
        self.image()
        self.scan()
        service = self.service()
        service.encoder_use_cpu = True
        service.ticket = 'cpu'
        service.execute('cpu')
        with self.assertRaisesRegex(ValueError, 'Create a new collection'):
            self.catalog.configure(use_cpu=False)
        self.assertTrue(FolderCatalog(self.catalog.directory).config['use_cpu'])

    def test_disabling_pixai_suspends_pending_predictions_without_losing_them(self):
        self.enable()
        self.image(prompt=None)
        self.scan()
        jobs = self.catalog.tag_pending(10)
        self.assertEqual(len(jobs), 1)
        self.catalog.configure(tagger='none')
        self.assertFalse(self.catalog.has_pending())
        self.assertEqual(self.catalog.tag_pending(10), [])
        self.assertEqual(self.catalog.status()['tagging'], {'pending': 1})
        self.catalog.configure(tagger='pixai')
        self.assertTrue(self.catalog.has_pending())
        self.assertEqual(self.catalog.tag_pending(10), jobs)

    def test_unpublished_embeddings_also_prevent_cpu_gpu_contract_change(self):
        self.image()
        self.scan()
        service = self.service()
        service.ticket = 'import'
        with patch.object(self.catalog, 'publish'):
            service.execute('import')
        self.assertEqual(self.catalog.status()['records'], {'staged': 1})
        with self.assertRaisesRegex(ValueError, 'Create a new collection'):
            self.catalog.configure(use_cpu=True)


class RuntimePromptTests(unittest.TestCase):
    def test_save_is_branch_local_and_roundtrips_actual_text(self):
        saved = []
        class Save:
            def save_images(self, images, prefix, graph, extra):
                saved.append(extra)
                return {"ui": {"images": []}}
        source = {"workflow": {"nodes": []}}
        with patch.dict(sys.modules, {"nodes": SimpleNamespace(SaveImage=Save)}):
            node = VRSSaveImage()
            result = node.save("pixels", "1girl, blue hair", negative="bad hands", extra_pnginfo=source)
            self.assertEqual(result["result"], ("pixels",))
            node.save("pixels", "1boy, red hair", extra_pnginfo=source)
        self.assertNotIn("rs_vrs_prompt", source)
        metadata = extract({k: json.dumps(v) for k, v in saved[0].items()}, "out.png")
        self.assertEqual(metadata["positive"], ["1girl, blue hair"])
        self.assertEqual(metadata["negative"], ["bad hands"])
        self.assertEqual(extract(saved[1])["positive"], ["1boy, red hair"])
        self.assertEqual(metadata["runtime_source"], "VRS Save Image")

    def test_unknown_runtime_schema_never_claims_valid_prompt(self):
        self.assertIsNone(extract({"rs_vrs_prompt": {"version": 99, "positive": "invented"}}))
        with self.assertRaises(ValueError):
            VRSSaveImage().save("pixels", "")


class AutoResumeTests(unittest.TestCase):
    setUp = fixtures.FolderTests.setUp

    def test_restart_preserves_opt_in_but_explicit_stop_clears_it(self):
        from vrs_folder_service import FolderService
        from vrs_int.vision_assets import AUTO_VISION
        from unittest.mock import AsyncMock
        self.catalog.config.update(resume_on_start=True, model_name=AUTO_VISION)
        (self.catalog.directory / "config.json").write_text(json.dumps(self.catalog.config), encoding="utf-8")
        async def run():
            service = FolderService(SimpleNamespace(prompt_queue=fixtures.Queue()), self.root / "collections")
            service.run = AsyncMock()
            service.start({"collection_id": self.catalog.config["id"]})
            await service.task
            service.shutdown()
            self.assertTrue((service.root / "resume.json").is_file())
            restarted = FolderService(service.server, service.root)
            restarted.run = AsyncMock()
            self.assertIsNotNone(restarted.resume_saved())
            await restarted.task
            self.assertEqual(restarted.catalog.config["id"], self.catalog.config["id"])
            restarted.stop()
            self.assertFalse((service.root / "resume.json").exists())
            self.assertIsNone(FolderService(service.server, service.root).resume_saved())
        with patch.dict(sys.modules, {"folder_paths": SimpleNamespace()}):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
