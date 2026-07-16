from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import argparse
import json
import mimetypes
import os
import signal
import shutil
import threading
from typing import Any

from . import __version__
from .adapters import adapter_catalog, create_adapter, jpeg_bytes
from .conversion import camera_feature_map, preview_output_frame, revision_catalog
from .manager import JobManager
from .models import DatasetDescriptor, EpisodeRef, JobConfig


STATIC_DIR = Path(__file__).with_name("static")


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], manager: JobManager):
        super().__init__(address, RequestHandler)
        self.manager = manager


class RequestHandler(BaseHTTPRequestHandler):
    server: AppServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._json({"ok": True, "version": __version__})
            elif parsed.path == "/api/bootstrap":
                self._json(
                    {
                        "version": __version__,
                        "adapters": adapter_catalog(),
                        "revisions": revision_catalog(),
                        "hardware": self.server.manager.hardware(),
                        "jobs": self.server.manager.list_jobs(),
                    }
                )
            elif parsed.path == "/api/jobs":
                self._json({"jobs": self.server.manager.list_jobs()})
            elif parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/preview"):
                job_id = parsed.path.split("/")[3]
                self._job_preview(job_id, parse_qs(parsed.query))
            elif parsed.path.startswith("/api/jobs/"):
                self._json(self.server.manager.get_job(parsed.path.split("/")[3]))
            elif parsed.path == "/api/fs":
                query = parse_qs(parsed.query)
                self._json(_list_directory(query.get("path", [str(Path.home())])[0]))
            elif parsed.path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "API endpoint not found")
            else:
                self._static(parsed.path)
        except Exception as exc:
            self._handle_exception(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/inspect":
                descriptor = self.server.manager.inspect_source(
                    payload["adapter"], payload["source_path"], payload.get("adapter_options")
                )
                self._json(descriptor)
            elif parsed.path == "/api/motion-scan":
                self._json(self.server.manager.scan_motion(payload))
            elif parsed.path == "/api/preview/raw":
                adapter = create_adapter(
                    payload["adapter"], payload["source_path"], payload.get("adapter_options") or {}
                )
                episode = EpisodeRef.from_dict(payload["episode"])
                image = adapter.preview(episode, payload["camera"], int(payload.get("frame_index", 0)))
                self._bytes(jpeg_bytes(image), "image/jpeg", cache=False)
            elif parsed.path == "/api/jobs":
                self._json(self.server.manager.create_job(payload), status=HTTPStatus.CREATED)
            elif parsed.path == "/api/jobs/resume-path":
                self._json(self.server.manager.resume_from_output(payload["output_path"]))
            elif parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/stop"):
                self._json(self.server.manager.stop_job(parsed.path.split("/")[3]))
            elif parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/resume"):
                self._json(self.server.manager.resume_job(parsed.path.split("/")[3]))
            else:
                self._error(HTTPStatus.NOT_FOUND, "API endpoint not found")
        except Exception as exc:
            self._handle_exception(exc)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/jobs/"):
                query = parse_qs(parsed.query)
                self.server.manager.delete_job(
                    parsed.path.split("/")[3], query.get("remove_cache", ["0"])[0] == "1"
                )
                self._json({"ok": True})
            else:
                self._error(HTTPStatus.NOT_FOUND, "API endpoint not found")
        except Exception as exc:
            self._handle_exception(exc)

    def _job_preview(self, job_id: str, query: dict[str, list[str]]) -> None:
        record = self.server.manager.get_job(job_id)
        config = JobConfig.from_dict(record["config"])
        descriptor = DatasetDescriptor.from_dict(record["descriptor"])
        episode_index = int(query.get("episode", ["0"])[0])
        frame_index = int(query.get("frame", ["0"])[0])
        raw_camera = query.get("camera", [descriptor.cameras[0]])[0]
        kind = query.get("kind", ["raw"])[0]
        if not 0 <= episode_index < len(descriptor.episodes):
            raise IndexError(episode_index)

        if kind == "raw":
            adapter = create_adapter(config.adapter, config.source_path, config.adapter_options)
            image = adapter.preview(descriptor.episodes[episode_index], raw_camera, frame_index)
        else:
            feature_map = camera_feature_map(config, descriptor)
            if raw_camera not in feature_map:
                raise ValueError(f"Camera field is not mapped to LeRobot output: {raw_camera}")
            feature_key = feature_map[raw_camera]
            final_root = Path(config.output_path)
            if record["state"] == "completed" and final_root.exists():
                image = preview_output_frame(
                    final_root, config.revision, feature_key, episode_index, frame_index
                )
            else:
                manifest = _read_json(Path(record["cache_dir"]) / "manifest.json")
                segment = next(
                    (
                        item
                        for item in manifest["segments"]
                        if item["status"] == "done" and item["start"] <= episode_index < item["end"]
                    ),
                    None,
                )
                if segment is None:
                    raise FileNotFoundError("This episode has not completed conversion yet")
                segment_root = Path(record["cache_dir"]) / "segments" / f"segment-{segment['id']}"
                image = preview_output_frame(
                    segment_root, "v2.1", feature_key, episode_index - segment["start"], frame_index
                )
        self._bytes(jpeg_bytes(image), "image/jpeg", cache=False)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2 * 1024 * 1024:
            raise ValueError("A JSON request body is required")
        return json.loads(self.rfile.read(length))

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._bytes(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
            cache=False,
        )

    def _bytes(
        self,
        value: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(value)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
            self._error(HTTPStatus.FORBIDDEN, "Invalid path")
            return
        if not path.is_file():
            if "." not in Path(relative).name:
                path = STATIC_DIR / "index.html"
            else:
                self._error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".js", ".css", ".webmanifest"}:
            content_type += "; charset=utf-8"
        self._bytes(path.read_bytes(), content_type, cache=path.name != "sw.js")

    def _handle_exception(self, exc: Exception) -> None:
        if isinstance(exc, KeyError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(exc, (ValueError, FileExistsError, FileNotFoundError, IndexError)):
            status = HTTPStatus.BAD_REQUEST
        elif isinstance(exc, PermissionError):
            status = HTTPStatus.FORBIDDEN
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._error(status, str(exc) or type(exc).__name__)

    def _error(self, status: HTTPStatus, message: str) -> None:
        if self.wfile.closed:
            return
        self._json({"error": message, "status": int(status)}, status=status)

    def log_message(self, format_string: str, *args: Any) -> None:
        if os.environ.get("LEROBOT_DATACONVERT_HTTP_LOG") == "1":
            super().log_message(format_string, *args)


def _list_directory(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)
    entries = []
    try:
        children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except PermissionError:
        raise
    for child in children[:500]:
        try:
            entry = {
                "name": child.name,
                "path": str(child),
                "kind": "directory" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            }
            entries.append(entry)
        except (PermissionError, FileNotFoundError):
            continue
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "parent": str(path.parent),
        "entries": entries,
        "free_bytes": usage.free,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot Dataset Convert workbench")
    parser.add_argument("--host", default=os.environ.get("LEROBOT_DATACONVERT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LEROBOT_DATACONVERT_PORT", "8765")))
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("LEROBOT_DATACONVERT_STATE", Path.home() / ".local/share/lerobot-dataconvert")),
    )
    args = parser.parse_args()
    manager = JobManager(args.state_dir.expanduser())
    server = AppServer((args.host, args.port), manager)
    stopping = threading.Event()

    def request_shutdown(*_: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, name="http-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(f"LeRobot Data Convert running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.4)
    finally:
        manager.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
