from __future__ import annotations

from pathlib import Path
import json
import math
import multiprocessing
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any

import psutil

from .adapters import create_adapter
from .conversion import (
    SUPPORTED_REVISIONS,
    atomic_write_json,
    directory_size,
    run_finalize_worker,
    run_segment_worker,
)
from .models import DatasetDescriptor, JobConfig


TERMINAL_STATES = {"completed", "failed", "canceled"}
ACTIVE_STATES = {"queued", "running", "merging", "stopping"}


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    record TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def insert(self, record: dict[str, Any]) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs(id, state, record, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    record["id"],
                    record["state"],
                    json.dumps(record, ensure_ascii=False),
                    record["created_at"],
                    record["updated_at"],
                ),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT record FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def patch(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT record FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            record = json.loads(row[0])
            record.update(fields)
            record["updated_at"] = time.time()
            connection.execute(
                "UPDATE jobs SET state = ?, record = ?, updated_at = ? WHERE id = ?",
                (record["state"], json.dumps(record, ensure_ascii=False), record["updated_at"], job_id),
            )
        return record

    def list(self) -> list[dict[str, Any]]:
        with self.lock, self._connect() as connection:
            rows = connection.execute("SELECT record FROM jobs ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, job_id: str) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


class JobManager:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(state_dir / "jobs.sqlite3")
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._current_job_id: str | None = None
        self._current_processes: dict[str, multiprocessing.Process] = {}
        self._recover_interrupted_jobs()
        self._scheduler = threading.Thread(target=self._scheduler_loop, name="conversion-scheduler", daemon=True)
        self._scheduler.start()

    def hardware(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        affinity = _available_cpu_ids()
        return {
            "cpu_count": len(affinity),
            "cpu_ids": affinity,
            "memory_total_gb": round(memory.total / 1024**3, 1),
            "memory_available_gb": round(memory.available / 1024**3, 1),
        }

    def inspect_source(
        self, adapter_slug: str, source_path: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return create_adapter(adapter_slug, source_path, options or {}).inspect().to_dict()

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_path = str(Path(payload["source_path"]).expanduser().resolve())
        output_path = str(Path(payload["output_path"]).expanduser().resolve())
        adapter_options = dict(payload.get("adapter_options") or {})
        descriptor = create_adapter(payload["adapter"], source_path, adapter_options).inspect()
        config = self._normalize_config(payload, descriptor, source_path, output_path, adapter_options)
        self._validate_paths(config)

        for existing in self.store.list():
            if Path(existing["config"]["output_path"]) != Path(output_path):
                continue
            if existing["state"] in ACTIVE_STATES:
                raise FileExistsError(f"Another active job owns this output: {existing['id']}")
            if not config.overwrite:
                raise FileExistsError(f"A cached job already owns this output: {existing['id']}")
            self.store.patch(
                existing["id"],
                state="canceled",
                phase="canceled",
                message="Superseded by a new overwrite job",
                finished_at=time.time(),
            )

        cache_dir = cache_path_for_output(Path(output_path))
        if cache_dir.exists():
            if not config.overwrite:
                raise FileExistsError(f"Cache already exists. Resume it instead: {cache_dir}")
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)

        job_id = uuid.uuid4().hex[:12]
        segments = []
        for segment_number, start in enumerate(range(0, len(descriptor.episodes), config.segment_size)):
            end = min(start + config.segment_size, len(descriptor.episodes))
            segments.append(
                {
                    "id": f"{segment_number:06d}",
                    "start": start,
                    "end": end,
                    "source_indices": list(range(start, end)),
                    "status": "pending",
                    "attempts": 0,
                    "frames": 0,
                    "bytes": 0,
                    "duration_seconds": 0,
                }
            )
        now = time.time()
        manifest = {
            "version": 1,
            "job_id": job_id,
            "state": "queued",
            "config": config.to_dict(),
            "descriptor": descriptor.to_dict(),
            "segments": segments,
            "created_at": now,
            "updated_at": now,
            "active_seconds": 0.0,
        }
        atomic_write_json(cache_dir / "manifest.json", manifest)

        record = self._record_from_manifest(manifest, cache_dir)
        self.store.insert(record)
        self._wake.set()
        return record

    def resume_from_output(self, output_path: str) -> dict[str, Any]:
        output = Path(output_path).expanduser().resolve()
        cache_dir = cache_path_for_output(output)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No conversion cache for {output}")
        manifest = _read_json(manifest_path)
        job_id = manifest["job_id"]
        existing = self.store.get(job_id)
        if existing is None:
            existing = self._record_from_manifest(manifest, cache_dir)
            self.store.insert(existing)
        if manifest.get("state") == "completed" and output.exists():
            return self.store.patch(job_id, state="completed", phase="done", message="Output is complete")
        return self.resume_job(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return [self._public_record(record) for record in self.store.list()]

    def get_job(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return self._public_record(record)

    def stop_job(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record["state"] == "queued":
            return self.store.patch(job_id, state="paused", phase="paused", message="Paused before start")
        if record["state"] in {"running", "merging"}:
            record = self.store.patch(job_id, state="stopping", message="Stopping active segments")
        return self._public_record(record)

    def resume_job(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record["state"] == "completed":
            return self._public_record(record)
        manifest = self._prepare_manifest(Path(record["cache_dir"]), reset_failures=True)
        manifest["state"] = "queued"
        manifest["updated_at"] = time.time()
        atomic_write_json(Path(record["cache_dir"]) / "manifest.json", manifest)
        record = self.store.patch(
            job_id,
            state="queued",
            phase="queued",
            message="Queued from cache",
            error=None,
            traceback=None,
            finished_at=None,
        )
        self._wake.set()
        return self._public_record(record)

    def delete_job(self, job_id: str, remove_cache: bool = False) -> None:
        record = self.store.get(job_id)
        if record is None:
            return
        if record["state"] in ACTIVE_STATES:
            raise RuntimeError("Stop the job before deleting it")
        self.store.delete(job_id)
        if remove_cache:
            shutil.rmtree(record["cache_dir"], ignore_errors=True)

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._current_job_id:
            try:
                self.store.patch(self._current_job_id, state="stopping", message="Server is shutting down")
            except KeyError:
                pass
        self._wake.set()
        self._scheduler.join(timeout=15)
        for process in list(self._current_processes.values()):
            _terminate_process(process)

    def _normalize_config(
        self,
        payload: dict[str, Any],
        descriptor: DatasetDescriptor,
        source_path: str,
        output_path: str,
        adapter_options: dict[str, Any],
    ) -> JobConfig:
        revisions = {item["id"] for item in SUPPORTED_REVISIONS}
        revision = payload.get("revision", "v2.1")
        if revision not in revisions:
            raise ValueError(f"Unsupported revision: {revision}")
        hardware = self.hardware()
        cpu_cores = max(1, min(int(payload.get("cpu_cores", 4)), hardware["cpu_count"]))
        memory_gb = max(1.0, min(float(payload.get("memory_gb", 8)), hardware["memory_total_gb"]))
        camera_names = {
            camera: _safe_name((payload.get("camera_names") or {}).get(camera) or _default_camera_name(camera))
            for camera in descriptor.cameras
        }
        if len(set(camera_names.values())) != len(camera_names):
            raise ValueError("Camera output names must be unique")
        task_instruction = str(payload.get("task_instruction") or "").strip()
        if not task_instruction:
            raise ValueError("Task instruction is required")
        repo_id = _safe_repo_id(payload.get("repo_id") or Path(output_path).name)
        state_names = [str(value).strip() for value in payload.get("state_names", []) if str(value).strip()]
        action_names = [str(value).strip() for value in payload.get("action_names", []) if str(value).strip()]
        return JobConfig(
            adapter=payload["adapter"],
            source_path=source_path,
            output_path=output_path,
            revision=revision,
            repo_id=repo_id,
            robot_type=str(payload.get("robot_type") or "unknown").strip(),
            task_instruction=task_instruction,
            fps=max(1, int(payload.get("fps") or descriptor.fps)),
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            segment_size=max(1, min(int(payload.get("segment_size", 2)), 16)),
            camera_names=camera_names,
            state_names=state_names,
            action_names=action_names,
            adapter_options={**adapter_options, "fps": max(1, int(payload.get("fps") or descriptor.fps))},
            skip_zero_state=bool(payload.get("skip_zero_state", True)),
            overwrite=bool(payload.get("overwrite", False)),
        )

    @staticmethod
    def _validate_paths(config: JobConfig) -> None:
        source = Path(config.source_path)
        output = Path(config.output_path)
        if not source.exists():
            raise FileNotFoundError(source)
        if source == output or (source.is_dir() and source in output.parents):
            raise ValueError("Output must not be inside the source dataset")
        if output.exists():
            nonempty = any(output.iterdir()) if output.is_dir() else True
            if nonempty and not config.overwrite:
                raise FileExistsError(f"Output already exists: {output}")
            if output.is_dir() and not nonempty:
                output.rmdir()
        output.parent.mkdir(parents=True, exist_ok=True)
        required = max(512 * 1024**2, int(_path_size(source) * 0.25))
        free = shutil.disk_usage(output.parent).free
        if free < required:
            raise OSError(f"Insufficient free space: need at least {required / 1024**3:.1f} GiB")

    def _scheduler_loop(self) -> None:
        while not self._shutdown.is_set():
            queued = [record for record in reversed(self.store.list()) if record["state"] == "queued"]
            if not queued:
                self._wake.wait(timeout=1)
                self._wake.clear()
                continue
            record = queued[0]
            self._current_job_id = record["id"]
            try:
                self._run_job(record)
            except BaseException as exc:
                self.store.patch(
                    record["id"],
                    state="failed",
                    phase="failed",
                    message="Scheduler failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=time.time(),
                )
            finally:
                self._current_job_id = None
                self._current_processes.clear()

    def _run_job(self, record: dict[str, Any]) -> None:
        job_id = record["id"]
        cache_dir = Path(record["cache_dir"])
        manifest = self._prepare_manifest(cache_dir)
        config = JobConfig.from_dict(manifest["config"])
        descriptor = DatasetDescriptor.from_dict(manifest["descriptor"])
        completed = [segment for segment in manifest["segments"] if segment["status"] == "done"]
        if len(completed) == len(manifest["segments"]):
            self._finalize_job(record, manifest)
            return

        run_started = time.monotonic()
        baseline_active = float(manifest.get("active_seconds", 0))
        self.store.patch(
            job_id,
            state="running",
            phase="convert",
            message="Converting independent segments",
            started_at=record.get("started_at") or time.time(),
            effective_workers=self._effective_workers(config, descriptor),
        )
        manifest["state"] = "running"
        atomic_write_json(cache_dir / "manifest.json", manifest)

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        allowed_cpus = _available_cpu_ids()[: config.cpu_cores]
        effective_workers = self._effective_workers(config, descriptor)
        active: dict[str, dict[str, Any]] = {}
        pending = [segment for segment in manifest["segments"] if segment["status"] != "done"]
        fatal_error: dict[str, Any] | None = None

        while pending or active:
            current = self.store.get(job_id)
            should_stop = self._shutdown.is_set() or current is None or current["state"] == "stopping"
            if should_stop:
                for item in active.values():
                    _terminate_process(item["process"])
                    item["segment"]["status"] = "pending"
                    shutil.rmtree(item["output_dir"], ignore_errors=True)
                manifest["active_seconds"] = baseline_active + (time.monotonic() - run_started)
                manifest["state"] = "paused"
                atomic_write_json(cache_dir / "manifest.json", manifest)
                self.store.patch(
                    job_id,
                    state="paused",
                    phase="paused",
                    message="Stopped; incomplete segments were discarded",
                    active_seconds=manifest["active_seconds"],
                    active_frames=0,
                    memory_rss_mb=0,
                )
                return

            self._drain_segment_messages(result_queue, active, manifest, cache_dir)
            self._collect_dead_workers(active, pending, fatal_error_holder := {})
            fatal_error = fatal_error or fatal_error_holder.get("error")
            if fatal_error:
                break

            while pending and len(active) < effective_workers:
                segment = pending.pop(0)
                output_dir = cache_dir / "segments" / f"segment-{segment['id']}"
                shutil.rmtree(output_dir, ignore_errors=True)
                busy_cpus = {item["cpu_id"] for item in active.values()}
                cpu_id = next((value for value in allowed_cpus if value not in busy_cpus), allowed_cpus[0])
                segment["status"] = "running"
                segment["attempts"] = int(segment.get("attempts", 0)) + 1
                segment["started_at"] = time.time()
                payload = {
                    "segment_id": segment["id"],
                    "output_dir": str(output_dir),
                    "config": manifest["config"],
                    "descriptor": manifest["descriptor"],
                    "source_indices": segment["source_indices"],
                    "episodes": [
                        descriptor.episodes[index].to_dict() for index in segment["source_indices"]
                    ],
                    "cpu_id": cpu_id,
                }
                process = context.Process(target=run_segment_worker, args=(payload, result_queue), daemon=True)
                process.start()
                active[segment["id"]] = {
                    "process": process,
                    "segment": segment,
                    "output_dir": output_dir,
                    "cpu_id": cpu_id,
                    "frames": 0,
                    "terminal": False,
                }
                self._current_processes[segment["id"]] = process
                atomic_write_json(cache_dir / "manifest.json", manifest)
                time.sleep(0.08)

            elapsed = baseline_active + (time.monotonic() - run_started)
            self._update_progress(job_id, manifest, descriptor, active, elapsed, effective_workers)
            time.sleep(0.2)

        if fatal_error:
            for item in active.values():
                _terminate_process(item["process"])
            manifest["active_seconds"] = baseline_active + (time.monotonic() - run_started)
            manifest["state"] = "failed"
            atomic_write_json(cache_dir / "manifest.json", manifest)
            self.store.patch(
                job_id,
                state="failed",
                phase="failed",
                message="A segment failed twice",
                error=fatal_error.get("error"),
                traceback=fatal_error.get("traceback"),
                active_seconds=manifest["active_seconds"],
                finished_at=time.time(),
            )
            return

        manifest["active_seconds"] = baseline_active + (time.monotonic() - run_started)
        manifest["state"] = "merging"
        atomic_write_json(cache_dir / "manifest.json", manifest)
        self._finalize_job(self.store.get(job_id) or record, manifest)

    def _drain_segment_messages(
        self,
        result_queue: Any,
        active: dict[str, dict[str, Any]],
        manifest: dict[str, Any],
        cache_dir: Path,
    ) -> None:
        while True:
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                break
            segment_id = message.get("segment_id")
            item = active.get(segment_id)
            if item is None:
                continue
            if message["type"] == "progress":
                item["frames"] = int(message["frames"])
                continue
            item["terminal"] = True
            process = item["process"]
            process.join(timeout=5)
            if message["type"] == "complete":
                self._current_processes.pop(segment_id, None)
                segment = item["segment"]
                segment.update(
                    {
                        "status": "done",
                        "frames": int(message["frames"]),
                        "bytes": int(message["bytes"]),
                        "duration_seconds": float(message["duration_seconds"]),
                        "peak_memory_mb": float(message["peak_memory_mb"]),
                        "finished_at": time.time(),
                    }
                )
                atomic_write_json(cache_dir / "manifest.json", manifest)
                active.pop(segment_id, None)
            else:
                item["segment"]["status"] = "failed"
                item["segment"]["last_error"] = message.get("error")
                item["segment"]["last_traceback"] = message.get("traceback")

    def _collect_dead_workers(
        self,
        active: dict[str, dict[str, Any]],
        pending: list[dict[str, Any]],
        holder: dict[str, Any],
    ) -> None:
        for segment_id, item in list(active.items()):
            process = item["process"]
            if process.is_alive():
                continue
            process.join(timeout=1)
            segment = item["segment"]
            marker_path = item["output_dir"] / ".segment-complete.json"
            if marker_path.exists():
                marker = _read_json(marker_path)
                segment.update(
                    {
                        "status": "done",
                        "frames": int(marker["frames"]),
                        "bytes": int(marker["bytes"]),
                        "duration_seconds": float(marker["duration_seconds"]),
                        "peak_memory_mb": float(marker.get("peak_memory_mb", 0)),
                    }
                )
                self._current_processes.pop(segment_id, None)
                active.pop(segment_id, None)
                continue
            if not item.get("terminal"):
                dead_seen_at = item.setdefault("dead_seen_at", time.monotonic())
                if time.monotonic() - dead_seen_at < 0.75:
                    continue
            if segment["status"] == "running":
                segment["status"] = "failed"
                segment["last_error"] = f"Worker exited with code {process.exitcode}"
            if segment["status"] == "failed":
                shutil.rmtree(item["output_dir"], ignore_errors=True)
                if int(segment.get("attempts", 0)) < 2:
                    segment["status"] = "pending"
                    pending.append(segment)
                else:
                    holder["error"] = {
                        "error": segment.get("last_error", "Worker failed"),
                        "traceback": segment.get("last_traceback"),
                    }
            self._current_processes.pop(segment_id, None)
            active.pop(segment_id, None)

    def _update_progress(
        self,
        job_id: str,
        manifest: dict[str, Any],
        descriptor: DatasetDescriptor,
        active: dict[str, dict[str, Any]],
        elapsed: float,
        effective_workers: int,
    ) -> None:
        completed = [segment for segment in manifest["segments"] if segment["status"] == "done"]
        completed_frames = sum(int(segment.get("frames", 0)) for segment in completed)
        active_frames = sum(int(item.get("frames", 0)) for item in active.values())
        progress_frames = completed_frames + active_frames
        total_frames = max(descriptor.total_frames, 1)
        fraction = min(progress_frames / total_frames, 0.995)
        written = sum(int(segment.get("bytes", 0)) for segment in completed)
        estimated = (
            int(written / completed_frames * descriptor.total_frames)
            if completed_frames > 0
            else int(descriptor.source_bytes * 0.4)
        )
        eta = max(0, elapsed / fraction - elapsed) if fraction > 0.002 else None
        memory = 0.0
        for item in active.values():
            try:
                memory += psutil.Process(item["process"].pid).memory_info().rss / 1024**2
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.store.patch(
            job_id,
            progress=round(fraction * 0.94, 5),
            completed_segments=len(completed),
            completed_episodes=sum(segment["end"] - segment["start"] for segment in completed),
            completed_frames=completed_frames,
            active_frames=active_frames,
            written_bytes=written,
            estimated_output_bytes=estimated,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            effective_workers=effective_workers,
            memory_rss_mb=round(memory, 1),
        )

    def _finalize_job(self, record: dict[str, Any], manifest: dict[str, Any]) -> None:
        job_id = record["id"]
        cache_dir = Path(record["cache_dir"])
        config = JobConfig.from_dict(manifest["config"])
        descriptor = DatasetDescriptor.from_dict(manifest["descriptor"])
        segment_dirs = [
            cache_dir / "segments" / f"segment-{segment['id']}"
            for segment in sorted(manifest["segments"], key=lambda value: value["start"])
        ]
        self.store.patch(
            job_id,
            state="merging",
            phase="merge" if config.revision == "v2.1" else "pack",
            message="Assembling the final LeRobot dataset",
            progress=max(float(record.get("progress", 0)), 0.94),
            eta_seconds=None,
        )
        manifest["state"] = "merging"
        atomic_write_json(cache_dir / "manifest.json", manifest)

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        payload = {
            "cache_dir": str(cache_dir),
            "config": manifest["config"],
            "descriptor": manifest["descriptor"],
            "segment_dirs": [str(path) for path in segment_dirs],
            "cpu_id": _available_cpu_ids()[0],
        }
        process = context.Process(target=run_finalize_worker, args=(payload, result_queue), daemon=True)
        process.start()
        self._current_processes["finalize"] = process
        final_message: dict[str, Any] | None = None
        while process.is_alive() or final_message is None:
            current = self.store.get(job_id)
            if self._shutdown.is_set() or (current and current["state"] == "stopping"):
                _terminate_process(process)
                shutil.rmtree(cache_dir / "assembled-v21", ignore_errors=True)
                shutil.rmtree(cache_dir / "assembled-v30", ignore_errors=True)
                manifest["state"] = "paused"
                atomic_write_json(cache_dir / "manifest.json", manifest)
                self.store.patch(
                    job_id,
                    state="paused",
                    phase="paused",
                    message="Stopped; incomplete assembly was discarded",
                    progress=0.94,
                )
                return
            try:
                message = result_queue.get(timeout=0.25)
                if message["type"] == "finalize_progress":
                    progress = 0.94 + float(message["progress"]) * 0.055
                    self.store.patch(job_id, phase=message["phase"], progress=progress)
                else:
                    final_message = message
            except queue.Empty:
                if not process.is_alive() and final_message is None:
                    final_message = {"type": "finalize_failed", "error": f"Finalizer exited with code {process.exitcode}"}
        process.join(timeout=5)
        self._current_processes.pop("finalize", None)

        if final_message["type"] != "finalized":
            manifest["state"] = "failed"
            atomic_write_json(cache_dir / "manifest.json", manifest)
            self.store.patch(
                job_id,
                state="failed",
                phase="failed",
                message="Final assembly failed",
                error=final_message.get("error"),
                traceback=final_message.get("traceback"),
                finished_at=time.time(),
            )
            return

        candidate = Path(final_message["candidate"])
        output = Path(config.output_path)
        if output.exists():
            if not config.overwrite:
                self.store.patch(
                    job_id,
                    state="failed",
                    phase="failed",
                    message="Output appeared while the task was running",
                    error=f"Output exists: {output}",
                    finished_at=time.time(),
                )
                return
            shutil.rmtree(output) if output.is_dir() else output.unlink()
        os.replace(candidate, output)
        result = final_message["result"]
        manifest.update(
            {
                "state": "completed",
                "updated_at": time.time(),
                "result": result,
                "output_path": str(output),
            }
        )
        atomic_write_json(cache_dir / "manifest.json", manifest)
        shutil.rmtree(cache_dir / "segments", ignore_errors=True)
        shutil.rmtree(cache_dir / "assembled-v21", ignore_errors=True)
        shutil.rmtree(cache_dir / "assembled-v30", ignore_errors=True)
        self.store.patch(
            job_id,
            state="completed",
            phase="done",
            message="Conversion completed",
            progress=1.0,
            completed_segments=len(manifest["segments"]),
            completed_episodes=len(descriptor.episodes),
            completed_frames=int(result["frames"]),
            active_frames=0,
            written_bytes=int(result["bytes"]),
            estimated_output_bytes=int(result["bytes"]),
            eta_seconds=0,
            memory_rss_mb=0,
            finished_at=time.time(),
        )

    def _effective_workers(self, config: JobConfig, descriptor: DatasetDescriptor) -> int:
        memory_workers = max(
            1, int(config.memory_gb * 1024 // max(descriptor.estimated_worker_memory_mb, 256))
        )
        return max(1, min(config.cpu_cores, memory_workers, len(descriptor.episodes)))

    def _prepare_manifest(self, cache_dir: Path, reset_failures: bool = False) -> dict[str, Any]:
        manifest_path = cache_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        for segment in manifest["segments"]:
            output_dir = cache_dir / "segments" / f"segment-{segment['id']}"
            marker_path = output_dir / ".segment-complete.json"
            if marker_path.exists():
                marker = _read_json(marker_path)
                segment.update(
                    {
                        "status": "done",
                        "frames": int(marker["frames"]),
                        "bytes": int(marker["bytes"]),
                        "duration_seconds": float(marker["duration_seconds"]),
                        "peak_memory_mb": float(marker.get("peak_memory_mb", 0)),
                    }
                )
            elif segment["status"] != "done" or manifest.get("state") != "completed":
                shutil.rmtree(output_dir, ignore_errors=True)
                segment["status"] = "pending"
                if reset_failures:
                    segment["attempts"] = 0
                    segment.pop("last_error", None)
                    segment.pop("last_traceback", None)
        manifest["updated_at"] = time.time()
        atomic_write_json(manifest_path, manifest)
        return manifest

    def _recover_interrupted_jobs(self) -> None:
        for record in self.store.list():
            if record["state"] in {"running", "merging", "stopping"}:
                cache_dir = Path(record["cache_dir"])
                if (cache_dir / "manifest.json").exists():
                    manifest = self._prepare_manifest(cache_dir)
                    manifest["state"] = "queued"
                    atomic_write_json(cache_dir / "manifest.json", manifest)
                    self.store.patch(
                        record["id"],
                        state="queued",
                        phase="queued",
                        message="Recovered after server interruption",
                        active_frames=0,
                        memory_rss_mb=0,
                    )

    def _record_from_manifest(self, manifest: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
        config = JobConfig.from_dict(manifest["config"])
        descriptor = DatasetDescriptor.from_dict(manifest["descriptor"])
        completed = [segment for segment in manifest["segments"] if segment["status"] == "done"]
        completed_frames = sum(int(segment.get("frames", 0)) for segment in completed)
        completed_episodes = sum(segment["end"] - segment["start"] for segment in completed)
        now = time.time()
        state = manifest.get("state", "queued")
        return {
            "id": manifest["job_id"],
            "name": config.repo_id,
            "state": state,
            "phase": "done" if state == "completed" else "queued",
            "message": "Restored from output cache" if completed else "Queued",
            "config": config.to_dict(),
            "descriptor": descriptor.to_dict(),
            "cache_dir": str(cache_dir),
            "created_at": float(manifest.get("created_at", now)),
            "updated_at": now,
            "started_at": None,
            "finished_at": now if state == "completed" else None,
            "active_seconds": float(manifest.get("active_seconds", 0)),
            "elapsed_seconds": float(manifest.get("active_seconds", 0)),
            "progress": 1.0 if state == "completed" else completed_frames / max(descriptor.total_frames, 1) * 0.94,
            "completed_segments": len(completed),
            "total_segments": len(manifest["segments"]),
            "completed_episodes": completed_episodes,
            "total_episodes": len(descriptor.episodes),
            "completed_frames": completed_frames,
            "total_frames": descriptor.total_frames,
            "active_frames": 0,
            "written_bytes": sum(int(segment.get("bytes", 0)) for segment in completed),
            "estimated_output_bytes": int(descriptor.source_bytes * 0.4),
            "eta_seconds": None,
            "effective_workers": self._effective_workers(config, descriptor),
            "memory_rss_mb": 0,
            "error": None,
            "traceback": None,
        }

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        output = dict(record)
        output.pop("traceback", None)
        return output


def cache_path_for_output(output: Path) -> Path:
    return output.parent / f".{output.name}.lerobot-cache"


def _available_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError(f"Invalid feature name: {value!r}")
    return cleaned


def _default_camera_name(camera: str) -> str:
    match = re.search(r"(\d+)$", camera)
    return f"image_{match.group(1)}" if match else camera


def _safe_repo_id(value: str) -> str:
    parts = [re.sub(r"[^a-zA-Z0-9_.-]+", "-", part).strip("-.") for part in str(value).split("/")]
    cleaned = "/".join(part for part in parts if part)
    if not cleaned:
        raise ValueError("Invalid repository id")
    return cleaned


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return directory_size(path)


def _terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=1)
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
