"""Folder discovery on CPU; bounded GPU batches on the normal ComfyUI queue.

No GPU background thread, queue worker patch, server restart or auto-download.
The service starts only after an explicit UI/API action, and pauses on restart.
"""
import asyncio
from itertools import islice
import json
import logging
from pathlib import Path
import threading
import time
import uuid

try:
    from .vrs_int.folders import FolderCatalog, collection_path
    from .vrs_int.generated import entry_from_record
    from .vrs_int.import_snapshot import file_digest
    from .vrs_int.native import NativeVision, load_image, pixel_digest
    from .vrs_int.store import Store
    from .vrs_int.vectors import encode_vector
except ImportError:
    from vrs_int.folders import FolderCatalog, collection_path
    from vrs_int.generated import entry_from_record
    from vrs_int.import_snapshot import file_digest
    from vrs_int.native import NativeVision, load_image, pixel_digest
    from vrs_int.store import Store
    from vrs_int.vectors import encode_vector


def _queue_view(queue):
    """Inspect immutable queue items without copying users' full workflows."""
    view = getattr(queue, "get_current_queue_volatile", None)
    return (view or queue.get_current_queue)()


def enqueue_if_idle(queue, item):
    """Atomic idle check + insertion relative to the ComfyUI prompt worker."""
    with queue.mutex:
        running, pending = _queue_view(queue)
        if running or pending:
            return False
        queue.put(item)
        return True


def generation_waiting(queue, ticket):
    """A normal prompt admitted after our batch gets the next image boundary."""
    with queue.mutex:
        running, pending = _queue_view(queue)
        return any(item[1] != ticket for item in running + pending)


class SearchPreparationYield(Exception):
    """CPU preparation yields to user generation or an explicit collection stop."""


class FolderService:
    def __init__(self, server, root):
        self.server, self.root = server, Path(root)
        self.catalog = None
        self.task = None
        self.stopped = threading.Event()
        self.stopped.set()
        self.state = "stopped"
        self.error = None
        self.ticket = None
        self.encoder = None
        self.encoder_name = None
        self.scan_count = 0
        self.guard = threading.RLock()
        self.prepared_snapshot = None
        self.search_error = None
        self.search_retry_at = 0

    def collections(self):
        result = []
        for path in sorted(self.root.glob("*/config.json")):
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
                collection_path(self.root, config["id"])
                result.append(config)
            except (ValueError, KeyError, OSError):
                continue
        return result

    def open(self, collection_id):
        return FolderCatalog(collection_path(self.root, collection_id))

    def status(self):
        return dict(state=self.state, error=self.error, active=not self.stopped.is_set(),
                    search_error=self.search_error,
                    scan_count=self.scan_count, ticket=self.ticket,
                    collection=self.catalog.status() if self.catalog else None,
                    collections=self.collections())

    def start(self, config):
        if (self.task and not self.task.done()) or self.ticket:
            raise ValueError("Stop the current import and wait for its batch to finish first")
        import folder_paths
        if config.get("collection_id"):
            catalog = self.open(config["collection_id"])
        else:
            roots = config.get("roots", [])
            if not isinstance(roots, list) or any(not isinstance(p, str) for p in roots):
                raise ValueError("roots must be a list of server-side folder paths")
            roots = list(roots)
            if config.get("include_output", False):
                roots.append(folder_paths.get_output_directory())
            model_name = config.get("model_name", "")
            model = folder_paths.get_full_path_or_raise("text_encoders", model_name)
            if Path(model).suffix != ".safetensors":
                raise ValueError("Select native Gemma safetensors for VRS vision")
            catalog = FolderCatalog.create(self.root, roots, model_name,
                label=config.get("label", "Personal images"), batch_size=config.get("batch_size", 4))
        folder_paths.get_full_path_or_raise("text_encoders", catalog.config["model_name"])
        catalog.retry()
        self.catalog = catalog
        self.error = None
        self.search_error = None
        self.prepared_snapshot = None
        self.search_retry_at = 0
        self.scan_count = 0
        self.stopped.clear()
        self.state = "scanning"
        self.task = asyncio.create_task(self.run())
        return self.status()

    def stop(self):
        self.stopped.set()
        with self.guard:
            ticket = self.ticket
            if ticket and self.server.prompt_queue.delete_queue_item(lambda item: item[1] == ticket):
                self.ticket = None
        self.state = "stopping" if self.ticket or (self.task and not self.task.done()) else "stopped"
        # Do not send ComfyUI's global interrupt: it might stop a user generation.
        return self.status()

    async def submit(self):
        import execution
        ticket = str(uuid.uuid4())
        prompt = {"1": {"class_type": "VRSFolderImportBatch", "inputs": {"ticket": ticket}}}
        valid = await execution.validate_prompt(ticket, prompt, None)
        if not valid[0]:
            raise ValueError(f"Folder batch prompt is invalid: {valid[1]}")
        # Finite, JSON-safe priority: user submissions remain ahead of an import
        # that was admitted while idle but has not started yet.
        item = (1e15, ticket, prompt,
                {"create_time": int(time.time() * 1000), "vrs_folder_import": True,
                 "client_id": "rs-vrs-folder-worker"}, valid[2], {})
        # A stop during async validation must not enqueue new work.
        with self.guard:
            if self.stopped.is_set():
                return
            self.ticket = ticket
            if enqueue_if_idle(self.server.prompt_queue, item):
                self.state = "queued"
            else:
                self.ticket = None
                self.state = "waiting-for-generation"

    async def run(self):
        scan = None
        next_scan = 0
        try:
            while not self.stopped.is_set():
                now = time.monotonic()
                if scan is None and now >= next_scan:
                    scan = self.catalog.scan_steps(self.stopped.is_set)
                if scan is not None:
                    chunk = await asyncio.to_thread(lambda: list(islice(scan, 32)))
                    self.scan_count += len(chunk)
                    if not chunk:
                        scan, next_scan = None, time.monotonic() + 30
                if self.stopped.is_set():
                    break
                ticket = self.ticket
                if ticket:
                    # Handles clearing pending work, validation/execution failure,
                    # and interrupts before the batch node was invoked.
                    with self.server.prompt_queue.mutex:
                        running, queued = _queue_view(self.server.prompt_queue)
                    if not any(item[1] == ticket for item in running + queued):
                        with self.guard:
                            if self.ticket == ticket:
                                self.ticket = None
                                raise RuntimeError("Import queue item disappeared before completion; resume to retry")
                else:
                    pending = await asyncio.to_thread(self.catalog.has_pending)
                    # Stop or an encoder error can occur while the DB is read.
                    if self.stopped.is_set():
                        break
                    await asyncio.to_thread(self.catalog.publish, force=not pending and scan is None)
                    await asyncio.to_thread(self.prepare_search)
                    if self.stopped.is_set():
                        break
                    if pending:
                        await self.submit()
                    else:
                        self.state = "scanning" if scan is not None else "watching"
                await asyncio.sleep(.05 if scan is not None else 1)
        except Exception as exc:
            self.error = str(exc)
            self.stopped.set()
            logging.exception("VRS folder import paused")
        finally:
            if scan is not None:
                scan.close()
            if not self.ticket and not self.error:
                try:
                    await asyncio.to_thread(self.catalog.publish, force=True)
                except Exception as exc:
                    self.error = str(exc)
                    self.stopped.set()
                    logging.exception("VRS folder publication paused")
            self.state = "stopping" if self.ticket else ("error" if self.error else "stopped")

    def prepare_search(self):
        """One CPU worker, only while idle; publish readiness after full success.

        This does no CUDA work and changes no global Torch thread settings.
        Stop/new prompts abort preparation at cooperative checkpoints. A failed
        preparation leaves the previous search head and ingestion intact.
        """
        if time.monotonic() < self.search_retry_at:
            return
        snapshot = self.catalog.snapshot()
        if snapshot is None or snapshot == self.prepared_snapshot:
            return
        def checkpoint():
            if self.stopped.is_set() or generation_waiting(self.server.prompt_queue, None):
                raise SearchPreparationYield()
        index = None
        try:
            checkpoint()
            self.state = "preparing-search"
            try:
                from .vrs_int.search_index import SearchIndex
            except ImportError:
                from vrs_int.search_index import SearchIndex
            index = SearchIndex(Store(self.catalog.memory), snapshot, checkpoint)
            checkpoint()
            if self.catalog.search_ready(snapshot):
                self.prepared_snapshot = snapshot
                self.search_error = None
        except SearchPreparationYield:
            self.state = "waiting-for-generation"
        except Exception as exc:
            self.search_error = str(exc)
            self.search_retry_at = time.monotonic() + 30
            logging.exception("VRS search preparation deferred; previous search head retained")
        finally:
            if index is not None:
                index.close()

    def execute(self, ticket):
        with self.guard:
            if ticket != self.ticket:
                raise ValueError("Expired folder import ticket; start/resume from the VRS folder panel")
        try:
            if self.stopped.is_set():
                self.catalog.publish(force=True)
                return {"state": "stopped", "added": 0}
            if generation_waiting(self.server.prompt_queue, ticket):
                return {"state": "yielded", "added": 0}
            self.state = "encoding"
            import folder_paths
            import comfy.model_management as mm
            name = self.catalog.config["model_name"]
            if self.encoder is None or self.encoder_name != name:
                self.encoder = NativeVision(folder_paths.get_full_path_or_raise("text_encoders", name))
                self.encoder_name = name
            # Loading/hash validation is synchronous, but a new user prompt
            # admitted during it must not wait for an additional image.
            if self.stopped.is_set() or generation_waiting(self.server.prompt_queue, ticket):
                if self.stopped.is_set():
                    self.catalog.publish(force=True)
                return {"state": "stopped" if self.stopped.is_set() else "yielded", "added": 0}
            encoder = self.encoder
            store = Store(self.catalog.memory)
            base = self.catalog.snapshot()
            contract = encoder.contract()
            if base and store.manifest(base)["contract"] != contract:
                raise ValueError("Vision model/runtime changed; create a new collection")
            added = 0
            for sha, path, row in self.catalog.pending(self.catalog.config["batch_size"]):
                if self.stopped.is_set() or generation_waiting(self.server.prompt_queue, ticket):
                    break
                mm.throw_exception_if_processing_interrupted()
                try:
                    if file_digest(path) != sha:
                        raise ValueError("Source changed since scan; waiting for rescan")
                    image = load_image(path)
                except (OSError, ValueError) as exc:
                    self.catalog.failed(sha, exc)
                    continue
                vector = encoder.encode_tensor(image)
                if file_digest(path) != sha:
                    self.catalog.failed(sha, "Source changed during encoding")
                    continue
                if hasattr(vector, "detach"):
                    vector = vector.detach().cpu().numpy()
                entry = entry_from_record(row, encode_vector(vector, contract["codec"]))
                entry["audit"].append({"native_pixel_sha256": pixel_digest(image)})
                self.catalog.stage([entry], [sha], contract)
                added += 1
            # Staging survives interruption. Let an arriving generation proceed
            # without a bucket rewrite; the CPU watcher publishes it next.
            if self.stopped.is_set() or not generation_waiting(self.server.prompt_queue, ticket):
                self.catalog.publish(force=self.stopped.is_set() or not self.catalog.has_pending())
            return dict(state="complete", added=added, snapshot=self.catalog.snapshot())
        except BaseException as exc:
            self.error = str(exc) or type(exc).__name__
            self.stopped.set()
            raise
        finally:
            with self.guard:
                self.ticket = None
            self.state = "error" if self.error else ("stopped" if self.stopped.is_set() else "watching")


_service = None


def get_service():
    global _service
    if _service is None:
        from server import PromptServer
        import folder_paths
        _service = FolderService(PromptServer.instance, Path(folder_paths.get_user_directory()) / "rs_vrs_folders")
    return _service


def register_routes():
    # Offline tools import this package too; they must not start a server/task.
    try:
        from server import PromptServer
    except ImportError:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(server, "_rs_vrs_folders_routes", False):
        return
    from aiohttp import web
    server._rs_vrs_folders_routes = True

    @server.routes.get("/rs-vrs/folders")
    async def status(request):
        import folder_paths
        value = await asyncio.to_thread(get_service().status)
        value["models"] = [n for n in folder_paths.get_filename_list("text_encoders") if n.endswith(".safetensors")]
        value["output_directory"] = folder_paths.get_output_directory()
        return web.json_response(value)

    @server.routes.post("/rs-vrs/folders/{action}")
    async def control(request):
        # Require JSON; a foreign HTML form cannot start folder reads. ComfyUI's
        # own origin/auth middleware remains responsible for server access.
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            action = request.match_info["action"]
            if action == "start":
                config = await request.json()
                if not isinstance(config, dict):
                    raise ValueError("Expected an object")
                value = get_service().start(config)
            elif action == "stop":
                value = get_service().stop()
            else:
                raise web.HTTPNotFound()
            return web.json_response(value)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)


class VRSFolderImportBatch:
    """Internal bounded work item; users operate the folder panel."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ticket": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "prompt/VRS/internal"
    OUTPUT_NODE = True
    DESCRIPTION = "Internal folder import batch. Start it from the VRS Folders panel."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, ticket):
        report = json.dumps(get_service().execute(ticket), ensure_ascii=False)
        return {"ui": {"text": [report]}, "result": (report,)}


class VRSFolderMemory:
    @classmethod
    def INPUT_TYPES(cls):
        ids = [c["id"] for c in get_service().collections()]
        return {"required": {"collection_id": (ids or ["Create a collection in VRS Folders"],)}}

    RETURN_TYPES = ("VRS_MEMORY", "STRING", "STRING")
    RETURN_NAMES = ("memory", "snapshot", "report")
    FUNCTION = "load"
    CATEGORY = "prompt/VRS"
    DESCRIPTION = "Follow the latest search-ready folder snapshot. During preparation use the previous ready snapshot. Memory Loader with an exact hash always uses that snapshot."

    @classmethod
    def IS_CHANGED(cls, collection_id):
        return get_service().open(collection_id).search_snapshot()

    def load(self, collection_id):
        try:
            from .vrs_int.recommend import Recommender
        except ImportError:
            from vrs_int.recommend import Recommender
        catalog = get_service().open(collection_id)
        snapshot = catalog.search_snapshot()
        if snapshot is None:
            raise ValueError("No completed folder batch yet; wait for indexing in VRS Folders")
        return (Recommender(catalog.memory, snapshot), snapshot,
                json.dumps(dict(collection_id=collection_id, snapshot=snapshot, latest_snapshot=catalog.snapshot(),
                                awaiting_search=catalog.snapshot() != snapshot, store_path=str(catalog.memory))))
