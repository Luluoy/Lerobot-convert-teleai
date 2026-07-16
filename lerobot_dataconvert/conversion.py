from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import copy
import ctypes
import json
import math
import os
import resource
import shutil
import signal
import tempfile
import time
import traceback
from typing import Any, Callable

import cv2
import numpy as np

from .adapters import create_adapter
from .models import DatasetDescriptor, EpisodeRef, JobConfig


SUPPORTED_REVISIONS = (
    {"id": "v2.1", "label": "LeRobot v2.1", "description": "One Parquet and one video per episode."},
    {"id": "v3.0", "label": "LeRobot v3.0", "description": "Packed Parquet and video files with episode offsets."},
)


def revision_catalog() -> list[dict[str, str]]:
    return [dict(item) for item in SUPPORTED_REVISIONS]


def camera_feature_map(config: JobConfig, descriptor: DatasetDescriptor) -> dict[str, str]:
    return {
        camera: f"observation.images.{config.camera_names.get(camera, camera)}"
        for camera in descriptor.cameras
    }


def make_features(config: JobConfig, descriptor: DatasetDescriptor) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for camera, feature_key in camera_feature_map(config, descriptor).items():
        features[feature_key] = {
            "dtype": "video",
            "shape": descriptor.camera_shapes[camera],
            "names": ["height", "width", "channels"],
        }

    state_names = _dimension_names(config.state_names, descriptor.state_dim, "state")
    action_names = _dimension_names(config.action_names, descriptor.action_dim, "action")
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (descriptor.state_dim,),
        "names": {"motors": state_names},
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (descriptor.action_dim,),
        "names": {"motors": action_names},
    }
    return features


def run_segment_worker(payload: dict[str, Any], result_queue: Any) -> None:
    started = time.monotonic()
    output_dir = Path(payload["output_dir"])
    segment_id = payload["segment_id"]
    try:
        _configure_worker(payload.get("cpu_id"))
        if output_dir.exists():
            shutil.rmtree(output_dir)

        config = JobConfig.from_dict(payload["config"])
        descriptor = DatasetDescriptor.from_dict(payload["descriptor"])
        episodes = [EpisodeRef.from_dict(item) for item in payload["episodes"]]
        source_indices = payload["source_indices"]
        adapter = create_adapter(config.adapter, config.source_path, config.adapter_options)

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=f"{config.repo_id}_segment_{segment_id}",
            root=str(output_dir),
            robot_type=config.robot_type,
            fps=config.fps,
            features=make_features(config, descriptor),
            image_writer_threads=0,
            image_writer_processes=0,
        )

        feature_map = camera_feature_map(config, descriptor)
        total_frames = 0
        episode_lengths: list[int] = []
        source_records: list[dict[str, Any]] = []
        try:
            for local_episode_index, (source_index, episode) in enumerate(zip(source_indices, episodes, strict=True)):
                episode_frames = 0
                for sample in adapter.iter_frames(episode):
                    state = np.asarray(sample.state, dtype=np.float32).reshape(-1)
                    action = np.asarray(sample.action, dtype=np.float32).reshape(-1)
                    if state.size != descriptor.state_dim or action.size != descriptor.action_dim:
                        raise ValueError(
                            f"Dimension mismatch in {episode.path}: state={state.size}, action={action.size}"
                        )
                    if config.skip_zero_state and np.all(state == 0):
                        continue

                    frame: dict[str, Any] = {"observation.state": state, "action": action}
                    for camera, feature_key in feature_map.items():
                        image = sample.images.get(camera)
                        if image is None:
                            raise ValueError(f"Missing or invalid {camera} frame in {episode.path}")
                        expected_h, expected_w, _ = descriptor.camera_shapes[camera]
                        if image.shape[:2] != (expected_h, expected_w):
                            image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
                        frame[feature_key] = np.ascontiguousarray(image, dtype=np.uint8)

                    dataset.add_frame(frame, config.task_instruction)
                    episode_frames += 1
                    total_frames += 1
                    if total_frames % 10 == 0:
                        result_queue.put(
                            {
                                "type": "progress",
                                "segment_id": segment_id,
                                "frames": total_frames,
                                "episode": source_index,
                            }
                        )

                if episode_frames <= 0:
                    raise ValueError(f"Episode produced no valid frames: {episode.path}")
                dataset.save_episode()
                episode_lengths.append(episode_frames)
                source_records.append(
                    {
                        "episode_index": local_episode_index,
                        "source_index": source_index,
                        "source_key": episode.key,
                        "source_path": episode.path,
                        "length": episode_frames,
                    }
                )
        finally:
            _close_dataset(dataset)

        _write_jsonl(output_dir / "meta" / "source_paths.jsonl", source_records)
        marker = {
            "segment_id": segment_id,
            "episodes": len(episode_lengths),
            "frames": total_frames,
            "episode_lengths": episode_lengths,
            "source_indices": source_indices,
            "bytes": directory_size(output_dir),
            "duration_seconds": time.monotonic() - started,
            "peak_memory_mb": _peak_memory_mb(),
        }
        atomic_write_json(output_dir / ".segment-complete.json", marker)
        result_queue.put({"type": "complete", **marker})
    except BaseException as exc:
        result_queue.put(
            {
                "type": "failed",
                "segment_id": segment_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=30),
                "duration_seconds": time.monotonic() - started,
                "peak_memory_mb": _peak_memory_mb(),
            }
        )


def run_finalize_worker(payload: dict[str, Any], result_queue: Any) -> None:
    started = time.monotonic()
    cache_dir = Path(payload["cache_dir"])
    try:
        _configure_worker(payload.get("cpu_id"))
        config = JobConfig.from_dict(payload["config"])
        descriptor = DatasetDescriptor.from_dict(payload["descriptor"])
        segment_dirs = [Path(path) for path in payload["segment_dirs"]]
        assembled_v21 = cache_dir / "assembled-v21"
        assembled_v30 = cache_dir / "assembled-v30"
        shutil.rmtree(assembled_v21, ignore_errors=True)
        shutil.rmtree(assembled_v30, ignore_errors=True)

        result_queue.put({"type": "finalize_progress", "phase": "merge", "progress": 0.05})
        result = merge_v21_segments(segment_dirs, assembled_v21, config, descriptor)
        candidate = assembled_v21
        if config.revision == "v3.0":
            result_queue.put({"type": "finalize_progress", "phase": "pack", "progress": 0.62})
            result = convert_v21_to_v30(assembled_v21, assembled_v30)
            candidate = assembled_v30
        result_queue.put(
            {
                "type": "finalized",
                "candidate": str(candidate),
                "result": result,
                "duration_seconds": time.monotonic() - started,
                "peak_memory_mb": _peak_memory_mb(),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "type": "finalize_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=30),
                "duration_seconds": time.monotonic() - started,
            }
        )


def merge_v21_segments(
    segment_dirs: list[Path], output_dir: Path, config: JobConfig, descriptor: DatasetDescriptor
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True)
    if not segment_dirs:
        raise ValueError("No completed segments to merge")

    infos = [_read_json(path / "meta" / "info.json") for path in segment_dirs]
    template_info = infos[0]
    for path, info in zip(segment_dirs, infos, strict=True):
        if info.get("features") != template_info.get("features"):
            raise ValueError(f"Segment features differ: {path}")

    merged_info = copy.deepcopy(template_info)
    video_keys = [
        key for key, feature in merged_info["features"].items() if feature.get("dtype") == "video"
    ]
    global_tasks: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}
    task_maps: dict[Path, dict[int, int]] = {}
    for segment_dir in segment_dirs:
        mapping: dict[int, int] = {}
        tasks = sorted(_read_jsonl(segment_dir / "meta" / "tasks.jsonl"), key=lambda row: row["task_index"])
        for row in tasks:
            task = row["task"]
            if task not in task_to_index:
                task_to_index[task] = len(global_tasks)
                global_tasks.append({"task_index": task_to_index[task], "task": task})
            mapping[int(row["task_index"])] = task_to_index[task]
        task_maps[segment_dir] = mapping

    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    merged_sources: list[dict[str, Any]] = []
    episode_index = 0
    frame_offset = 0

    for segment_dir, segment_info in zip(segment_dirs, infos, strict=True):
        episodes = sorted(
            _read_jsonl(segment_dir / "meta" / "episodes.jsonl"), key=lambda row: row["episode_index"]
        )
        stats_by_episode = {
            int(row["episode_index"]): row for row in _read_jsonl(segment_dir / "meta" / "episodes_stats.jsonl")
        }
        sources_by_episode = {
            int(row["episode_index"]): row for row in _read_jsonl(segment_dir / "meta" / "source_paths.jsonl")
        }
        task_map = task_maps[segment_dir]

        for source_episode in episodes:
            local_index = int(source_episode["episode_index"])
            source_data = _v21_data_path(segment_dir, segment_info, local_index)
            table = pq.read_table(source_data)
            frame_count = table.num_rows

            def set_column(name: str, values: Any) -> None:
                nonlocal table
                array = pa.array(values)
                column_index = table.schema.get_field_index(name)
                table = table.append_column(name, array) if column_index < 0 else table.set_column(column_index, name, array)

            set_column("episode_index", np.full(frame_count, episode_index, dtype=np.int64))
            set_column("frame_index", np.arange(frame_count, dtype=np.int64))
            set_column("index", np.arange(frame_offset, frame_offset + frame_count, dtype=np.int64))
            if "task_index" in table.column_names:
                source_tasks = np.asarray(table["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                set_column("task_index", np.asarray([task_map.get(int(value), int(value)) for value in source_tasks]))

            destination_data = _v21_data_path(output_dir, merged_info, episode_index)
            destination_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination_data)

            for video_key in video_keys:
                source_video = _v21_video_path(segment_dir, segment_info, video_key, local_index)
                destination_video = _v21_video_path(output_dir, merged_info, video_key, episode_index)
                if not source_video.is_file():
                    raise FileNotFoundError(source_video)
                _link_or_copy(source_video, destination_video)

            merged_episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": list(source_episode.get("tasks", [])),
                    "length": frame_count,
                }
            )
            stats_row = stats_by_episode.get(local_index)
            if stats_row is None:
                raise FileNotFoundError(f"Missing stats for {segment_dir} episode {local_index}")
            merged_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": _remap_stats(
                        stats_row["stats"], episode_index, frame_offset, frame_count, task_map
                    ),
                }
            )
            source_row = sources_by_episode.get(local_index, {})
            merged_sources.append(
                {**source_row, "episode_index": episode_index, "length": frame_count}
            )
            episode_index += 1
            frame_offset += frame_count

    chunk_size = int(merged_info.get("chunks_size", 1000))
    merged_info.update(
        {
            "codebase_version": "v2.1",
            "total_episodes": episode_index,
            "total_frames": frame_offset,
            "total_tasks": len(global_tasks),
            "total_videos": episode_index * len(video_keys),
            "total_chunks": math.ceil(episode_index / chunk_size),
            "splits": {"train": f"0:{episode_index}"},
        }
    )
    atomic_write_json(output_dir / "meta" / "info.json", merged_info)
    _write_jsonl(output_dir / "meta" / "tasks.jsonl", global_tasks)
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", merged_episodes)
    _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", merged_stats)
    _write_jsonl(output_dir / "meta" / "source_paths.jsonl", merged_sources)
    atomic_write_json(
        output_dir / "meta" / "conversion.json",
        {
            "tool": "lerobot_dataconvert",
            "revision": "v2.1",
            "adapter": config.adapter,
            "source_path": config.source_path,
            "created_at": time.time(),
        },
    )
    return {"episodes": episode_index, "frames": frame_offset, "bytes": directory_size(output_dir)}


def convert_v21_to_v30(source_root: Path, output_root: Path) -> dict[str, Any]:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True)
    source_info = _read_json(source_root / "meta" / "info.json")
    episodes = sorted(_read_jsonl(source_root / "meta" / "episodes.jsonl"), key=lambda row: row["episode_index"])
    episode_stats = {
        int(row["episode_index"]): row["stats"]
        for row in _read_jsonl(source_root / "meta" / "episodes_stats.jsonl")
    }
    video_keys = [
        key for key, feature in source_info["features"].items() if feature.get("dtype") == "video"
    ]

    data_metadata: list[dict[str, Any]] = [{} for _ in episodes]
    data_groups = _group_episode_files(
        [_v21_data_path(source_root, source_info, int(ep["episode_index"])) for ep in episodes],
        limit_mb=100,
    )
    dataset_offset = 0
    episode_cursor = 0
    for file_number, group in enumerate(data_groups):
        chunk_index, file_index = divmod(file_number, 1000)
        tables = [pq.read_table(path) for path in group]
        destination = output_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(tables), destination)
        for table in tables:
            frame_count = table.num_rows
            data_metadata[episode_cursor] = {
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": dataset_offset,
                "dataset_to_index": dataset_offset + frame_count,
            }
            dataset_offset += frame_count
            episode_cursor += 1

    video_metadata: list[dict[str, Any]] = [defaultdict(dict) for _ in episodes]
    for video_key in sorted(video_keys):
        paths = [
            _v21_video_path(source_root, source_info, video_key, int(ep["episode_index"]))
            for ep in episodes
        ]
        groups = _group_episode_files(paths, limit_mb=200)
        episode_cursor = 0
        for file_number, group in enumerate(groups):
            chunk_index, file_index = divmod(file_number, 1000)
            destination = output_root / f"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            if len(group) == 1:
                _link_or_copy(group[0], destination)
            else:
                concatenate_videos(group, destination)
            timestamp = 0.0
            for _ in group:
                frame_count = int(episodes[episode_cursor]["length"])
                duration = frame_count / int(source_info["fps"])
                video_metadata[episode_cursor].update(
                    {
                        f"videos/{video_key}/chunk_index": chunk_index,
                        f"videos/{video_key}/file_index": file_index,
                        f"videos/{video_key}/from_timestamp": timestamp,
                        f"videos/{video_key}/to_timestamp": timestamp + duration,
                    }
                )
                timestamp += duration
                episode_cursor += 1

    tasks = sorted(_read_jsonl(source_root / "meta" / "tasks.jsonl"), key=lambda row: row["task_index"])
    task_frame = pd.DataFrame(
        {"task_index": [int(row["task_index"]) for row in tasks]},
        index=pd.Index([row["task"] for row in tasks], name="task"),
    )
    task_frame.to_parquet(output_root / "meta" / "tasks.parquet")

    episode_rows: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        row = {
            "episode_index": int(episode["episode_index"]),
            "tasks": list(episode.get("tasks", [])),
            "length": int(episode["length"]),
            **data_metadata[index],
            **video_metadata[index],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature, feature_stats in episode_stats[index].items():
            for statistic, value in feature_stats.items():
                row[f"stats/{feature}/{statistic}"] = value
        episode_rows.append(row)
    episode_path = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)

    aggregate = aggregate_episode_stats([episode_stats[index] for index in range(len(episodes))])
    atomic_write_json(output_root / "meta" / "stats.json", aggregate)

    info = copy.deepcopy(source_info)
    info["codebase_version"] = "v3.0"
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["chunks_size"] = 1000
    info["data_files_size_in_mb"] = 100
    info["video_files_size_in_mb"] = 200
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["fps"] = int(info["fps"])
    for feature in info["features"].values():
        if feature.get("dtype") != "video":
            feature["fps"] = info["fps"]
    atomic_write_json(output_root / "meta" / "info.json", info)
    source_records = source_root / "meta" / "source_paths.jsonl"
    if source_records.exists():
        shutil.copy2(source_records, output_root / "meta" / "source_paths.jsonl")
    conversion_meta = source_root / "meta" / "conversion.json"
    conversion = _read_json(conversion_meta) if conversion_meta.exists() else {}
    conversion["revision"] = "v3.0"
    conversion["packed_at"] = time.time()
    atomic_write_json(output_root / "meta" / "conversion.json", conversion)
    return {"episodes": len(episodes), "frames": dataset_offset, "bytes": directory_size(output_root)}


def concatenate_videos(paths: list[Path], destination: Path) -> None:
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    list_path: Path | None = None
    temporary_output: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as handle:
            handle.write("ffconcat version 1.0\n")
            for path in paths:
                escaped = str(path.resolve()).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
            list_path = Path(handle.name)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            temporary_output = Path(handle.name)

        source = av.open(str(list_path), "r", format="concat", options={"safe": "0"})
        output = av.open(str(temporary_output), "w", options={"movflags": "faststart"})
        stream_map: dict[int, Any] = {}
        for stream in source.streams:
            if stream.type in {"video", "audio", "subtitle"}:
                target = output.add_stream_from_template(stream, opaque=True)
                target.time_base = stream.time_base
                stream_map[stream.index] = target
        for packet in source.demux():
            if packet.dts is None or packet.stream.index not in stream_map:
                continue
            packet.stream = stream_map[packet.stream.index]
            output.mux(packet)
        source.close()
        output.close()
        shutil.move(str(temporary_output), destination)
        temporary_output = None
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def preview_output_frame(
    root: Path, revision: str, camera_key: str, episode_index: int, frame_index: int
) -> np.ndarray:
    import av
    import pyarrow.parquet as pq

    info = _read_json(root / "meta" / "info.json")
    fps = int(info["fps"])
    if revision == "v2.1":
        video_path = _v21_video_path(root, info, camera_key, episode_index)
        timestamp = frame_index / fps
    elif revision == "v3.0":
        episode_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
        table = pq.read_table(episode_files)
        indices = np.asarray(table["episode_index"].to_numpy())
        matches = np.flatnonzero(indices == episode_index)
        if not matches.size:
            raise IndexError(episode_index)
        row = int(matches[0])
        chunk = int(table[f"videos/{camera_key}/chunk_index"][row].as_py())
        file_index = int(table[f"videos/{camera_key}/file_index"][row].as_py())
        start = float(table[f"videos/{camera_key}/from_timestamp"][row].as_py())
        timestamp = start + frame_index / fps
        video_path = root / info["video_path"].format(
            video_key=camera_key, chunk_index=chunk, file_index=file_index
        )
    else:
        raise ValueError(f"Unsupported revision: {revision}")

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        seek_target = max(0, int(timestamp / float(stream.time_base)))
        container.seek(seek_target, stream=stream, any_frame=False, backward=True)
        best = None
        for frame in container.decode(stream):
            best = frame
            if frame.time is not None and frame.time + (0.5 / fps) >= timestamp:
                break
        if best is None:
            raise ValueError(f"No frame decoded from {video_path}")
        return best.to_ndarray(format="rgb24")


def aggregate_episode_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    features = sorted({key for episode in episodes for key in episode})
    for feature in features:
        parts = [episode[feature] for episode in episodes if feature in episode]
        if not parts:
            continue
        counts = np.asarray([float(np.asarray(part["count"]).reshape(-1)[0]) for part in parts])
        means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
        stds = [np.asarray(part["std"], dtype=np.float64) for part in parts]
        total = float(counts.sum())
        mean = sum(value * count for value, count in zip(means, counts, strict=True)) / max(total, 1.0)
        variance = sum(
            count * (std**2 + (part_mean - mean) ** 2)
            for part_mean, std, count in zip(means, stds, counts, strict=True)
        ) / max(total, 1.0)
        aggregated[feature] = {
            "min": np.minimum.reduce([np.asarray(part["min"]) for part in parts]).tolist(),
            "max": np.maximum.reduce([np.asarray(part["max"]) for part in parts]).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(variance, 0)).tolist(),
            "count": [int(total)],
        }
    return aggregated


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _configure_worker(cpu_id: int | None) -> None:
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    cv2.setNumThreads(1)
    if cpu_id is not None:
        try:
            os.sched_setaffinity(0, {int(cpu_id)})
        except (AttributeError, OSError):
            pass
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGTERM)
    except (OSError, AttributeError):
        pass


def _close_dataset(dataset: Any) -> None:
    if getattr(dataset, "image_writer", None) is not None and hasattr(dataset, "_wait_image_writer"):
        dataset._wait_image_writer()
    if getattr(dataset, "image_writer", None) is not None and hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def _dimension_names(names: list[str], dimension: int, prefix: str) -> list[str]:
    return names if len(names) == dimension else [f"{prefix}_{index}" for index in range(dimension)]


def _peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / 1024, 1)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _v21_chunk(info: dict[str, Any], episode_index: int) -> int:
    return episode_index // int(info.get("chunks_size", 1000))


def _v21_data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=_v21_chunk(info, episode_index), episode_index=episode_index
    )


def _v21_video_path(root: Path, info: dict[str, Any], video_key: str, episode_index: int) -> Path:
    return root / info["video_path"].format(
        episode_chunk=_v21_chunk(info, episode_index), video_key=video_key, episode_index=episode_index
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _range_stats(start: int, count: int) -> dict[str, list[Any]]:
    values = np.arange(start, start + count, dtype=np.float64)
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [count],
    }


def _constant_stats(value: int, count: int) -> dict[str, list[Any]]:
    return {"min": [value], "max": [value], "mean": [float(value)], "std": [0.0], "count": [count]}


def _remap_stats(
    stats: dict[str, Any], episode_index: int, frame_offset: int, frame_count: int, task_map: dict[int, int]
) -> dict[str, Any]:
    output = copy.deepcopy(stats)
    output["frame_index"] = _range_stats(0, frame_count)
    output["index"] = _range_stats(frame_offset, frame_count)
    output["episode_index"] = _constant_stats(episode_index, frame_count)
    task_stats = output.get("task_index")
    if task_stats:
        source_task = int(round(float(np.asarray(task_stats["mean"]).reshape(-1)[0])))
        output["task_index"] = _constant_stats(task_map.get(source_task, source_task), frame_count)
    return output


def _group_episode_files(paths: list[Path], limit_mb: int) -> list[list[Path]]:
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    limit = limit_mb * 1024 * 1024
    for path in paths:
        size = path.stat().st_size
        if current and current_bytes + size > limit:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
    if current:
        groups.append(current)
    return groups
