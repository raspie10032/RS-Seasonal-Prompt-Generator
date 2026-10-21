"""Read-only setup checks stay bounded and do not install, download or infer."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrs_int.folder_preview import preview_folders, sample_folder
from vrs_int.folders import normalize_roots
from vrs_int.vision_assets import AUTO_VISION
import test_vrs_folders as fixtures


class FolderPreviewTests(unittest.TestCase):
    setUp = fixtures.FolderTests.setUp
    image = fixtures.FolderTests.image

    def test_sample_metadata_missing_and_corrupt_images_without_changes(self):
        self.image()
        self.image("plain.png", prompt=None)
        (self.images / "broken.png").write_bytes(b"not an image")
        before = {p.name: p.read_bytes() for p in self.images.iterdir()}
        with patch("socket.socket", side_effect=AssertionError("No network")):
            result = sample_folder(self.images)
        self.assertEqual((result["images_checked"], result["readable_metadata"],
                          result["no_metadata"], result["read_errors"]), (3, 1, 1, 1))
        self.assertTrue(result["complete"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.images.iterdir()})

    def test_entry_budget_also_bounds_folders_without_images(self):
        for i in range(20):
            (self.images / f"{i}.txt").touch()
        result = sample_folder(self.images, entry_limit=3)
        self.assertEqual(result["entries_checked"], 3)
        self.assertEqual(result["images_checked"], 0)
        self.assertTrue(result["limited"])

    def test_image_budget_and_cooperative_deadline(self):
        for i in range(5):
            self.image(f"{i}.png")
        self.assertEqual(sample_folder(self.images, image_limit=2)["images_checked"], 2)
        result = sample_folder(self.images, seconds=0)
        self.assertEqual(result["entries_checked"], 0)
        self.assertTrue(result["limited"])

    @unittest.skipIf(os.name == "nt", "Windows symlink creation needs privileges")
    def test_symlink_files_and_directories_are_skipped(self):
        outside = self.root / "outside"
        outside.mkdir()
        original = self.image()
        original.rename(outside / "original.png")
        (self.images / "link.png").symlink_to(outside / "original.png")
        (self.images / "linked").symlink_to(outside, target_is_directory=True)
        self.assertEqual(sample_folder(self.images)["images_checked"], 0)

    def test_normalization_matches_creation_and_does_not_create_output(self):
        nested = self.images / "nested"
        nested.mkdir()
        self.assertEqual(normalize_roots([str(nested), str(self.images), str(self.images)]), [str(self.images)])
        missing = self.root / "missing-output"
        with patch.dict(sys.modules, {"folder_paths": SimpleNamespace(get_output_directory=lambda: str(missing))}):
            with self.assertRaises(FileNotFoundError):
                preview_folders({"include_output": True})
        self.assertFalse(missing.exists())

    def test_pixai_missing_dependencies_block_start_without_catalog_writes(self):
        config = {"roots": [str(self.images)], "tagger": "pixai", "use_cpu": True}
        with patch.dict(sys.modules, {"folder_paths": SimpleNamespace()}), \
                patch("vrs_int.folder_preview.importlib.util.find_spec", return_value=None):
            result = preview_folders(config)
        self.assertEqual(result["missing_packages"], ["transformers", "timm"])
        self.assertFalse(result["can_start"])
        self.assertTrue(result["use_cpu"])
        self.assertEqual(result["model_name"], AUTO_VISION)

    def test_metadata_only_does_not_probe_optional_packages_or_models(self):
        with patch.dict(sys.modules, {"folder_paths": SimpleNamespace()}), \
                patch("vrs_int.folder_preview.importlib.util.find_spec", side_effect=AssertionError("Optional import")), \
                patch("vrs_int.folder_preview.local_vision_path", side_effect=AssertionError("Model read")):
            self.assertTrue(preview_folders({"roots": [str(self.images)]})["can_start"])

    def test_offline_flags_match_model_download_policy(self):
        for key in ("RS_MODELS_OFFLINE", "HF_HUB_OFFLINE"):
            with self.subTest(key=key), patch.dict(sys.modules, {"folder_paths": SimpleNamespace()}), \
                    patch.dict(os.environ, {"RS_MODELS_OFFLINE": "", "HF_HUB_OFFLINE": "", key: "ON"}):
                self.assertTrue(preview_folders({"roots": [str(self.images)]})["offline"])

    def test_preview_existing_collection_does_not_open_or_modify_catalog(self):
        from vrs_folder_service import FolderService
        service = FolderService(None, self.root / "collections")
        config_path = self.catalog.directory / "config.json"
        before = config_path.read_bytes()
        with patch.object(service, "open", side_effect=AssertionError("No DB open")), \
                patch("vrs_int.folder_preview.preview_folders", return_value={"ok": True}) as preview:
            self.assertEqual(service.preview({"collection_id": self.catalog.config["id"],
                                              "use_cpu": True, "roots": ["/ignored"]}), {"ok": True})
        config = preview.call_args.args[0]
        self.assertTrue(config["use_cpu"])
        self.assertEqual(config["roots"], [str(self.images)])
        self.assertEqual(config_path.read_bytes(), before)

    def test_invalid_options_rejected_before_sampling(self):
        with patch.dict(sys.modules, {"folder_paths": SimpleNamespace()}), \
                patch("vrs_int.folder_preview.sample_folder", side_effect=AssertionError("Do not scan")):
            for options in ({"batch_size": True}, {"batch_size": 33}, {"tagger": "other"},
                            {"use_cpu": "false"}, {"model_name": ""}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    preview_folders({"roots": [str(self.images)], **options})

    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "ComfyUI HTTP dependency is optional in offline tests")
    def test_preview_route_runs_off_event_loop_and_requires_json(self):
        from aiohttp import web
        import vrs_folder_service as module
        import threading
        routes = web.RouteTableDef()
        server = SimpleNamespace(routes=routes, app=web.Application())
        main_thread = threading.get_ident()
        observed = []
        class Service:
            def preview(self, config):
                observed.append(threading.get_ident())
                return {"can_start": True}
        with patch.dict(sys.modules, {"server": SimpleNamespace(PromptServer=SimpleNamespace(instance=server))}), \
                patch.object(module, "get_service", return_value=Service()):
            module.register_routes()
            control = next(route.handler for route in routes if route.method == "POST")
            from unittest.mock import AsyncMock
            request = SimpleNamespace(content_type="application/json", match_info={"action": "preview"},
                                      json=AsyncMock(return_value={"roots": [str(self.images)]}))
            response = asyncio.run(control(request))
            self.assertEqual(json.loads(response.body), {"can_start": True})
            self.assertNotEqual(observed, [main_thread])
            request.content_type = "application/x-www-form-urlencoded"
            with self.assertRaises(web.HTTPUnsupportedMediaType):
                asyncio.run(control(request))
            request.content_type = "application/json"
            request.json = AsyncMock(return_value=[])
            self.assertEqual(asyncio.run(control(request)).status, 400)

    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "ComfyUI HTTP dependency is optional in offline tests")
    def test_real_http_preview_preserves_files_and_never_creates_collection(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        import vrs_folder_service as module
        self.image()
        self.image("plain.png", prompt=None)
        routes = web.RouteTableDef()
        app = web.Application()
        server = SimpleNamespace(routes=routes, app=app)
        service = module.FolderService(server, self.root / "no-collections-created")
        before = {p.name: p.read_bytes() for p in self.images.iterdir()}

        async def check():
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/rs-vrs/folders/preview", json={"roots": [str(self.images)]})
                self.assertEqual(response.status, 200)
                result = await response.json()
                self.assertEqual(result["samples"][0]["readable_metadata"], 1)
                self.assertEqual(result["samples"][0]["no_metadata"], 1)
                self.assertTrue(result["can_start"])
                response = await client.post("/rs-vrs/folders/preview", json={"roots": [str(self.root / "absent")]})
                self.assertEqual(response.status, 400)
                response = await client.post("/rs-vrs/folders/preview", data={"roots": str(self.images)})
                self.assertEqual(response.status, 415)

        with patch.dict(sys.modules, {"server": SimpleNamespace(PromptServer=SimpleNamespace(instance=server)),
                                      "folder_paths": SimpleNamespace()}), \
                patch.object(module, "get_service", return_value=service):
            module.register_routes()
            app.add_routes(routes)
            asyncio.run(check())
        self.assertIsNone(service.task)
        self.assertFalse(service.root.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.images.iterdir()})


if __name__ == "__main__":
    unittest.main()
