from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .models import DatasetDescriptor, EpisodeRef


def analyze_action_sequence(
    values: Iterable[Mapping[str, Any]],
    action_fields: list[str],
    expected_frames: int,
    trim_stationary_start: bool,
    remove_stationary_segments: bool,
    stationary_frames: int,
) -> dict[str, Any]:
    if not action_fields:
        raise ValueError("The raw dataset adapter did not declare any action fields")

    runs: list[tuple[int, int]] = []
    previous: tuple[np.ndarray, ...] | None = None
    run_start = 0
    frame_count = 0
    for frame_count, frame_values in enumerate(values, start=1):
        missing = [name for name in action_fields if name not in frame_values]
        if missing:
            raise ValueError(f"Missing declared action fields: {', '.join(missing)}")
        current = tuple(np.asarray(frame_values[name]).copy() for name in action_fields)
        if previous is None:
            previous = current
            continue
        if not all(np.array_equal(left, right) for left, right in zip(previous, current, strict=True)):
            runs.append((run_start, frame_count - 1))
            run_start = frame_count - 1
        previous = current

    if frame_count != expected_frames:
        raise ValueError(f"Action scan produced {frame_count} frames; expected {expected_frames}")
    if frame_count == 0:
        raise ValueError("Action scan produced no frames")
    runs.append((run_start, frame_count))

    segments: list[dict[str, Any]] = []
    first_start, first_end = runs[0]
    if trim_stationary_start and first_end - first_start >= 2:
        end = first_end if first_end < frame_count else frame_count - 1
        if end > 0:
            segments.append({"start": 0, "end": end, "kind": "leading"})

    if remove_stationary_segments:
        minimum = max(2, int(stationary_frames))
        for index, (start, end) in enumerate(runs):
            if index == 0 and trim_stationary_start:
                continue
            if end - start < minimum:
                continue
            segments.append({"start": start + 1, "end": end, "kind": "stationary"})

    for segment in segments:
        segment["frames"] = segment["end"] - segment["start"]
    removed_frames = sum(int(segment["frames"]) for segment in segments)
    return {
        "source_frames": frame_count,
        "kept_frames": frame_count - removed_frames,
        "removed_frames": removed_frames,
        "segments": segments,
        "leading_segments": sum(segment["kind"] == "leading" for segment in segments),
        "stationary_segments": sum(segment["kind"] == "stationary" for segment in segments),
    }


def analyze_episode_motion(
    adapter: Any,
    descriptor: DatasetDescriptor,
    episode: EpisodeRef,
    trim_stationary_start: bool,
    remove_stationary_segments: bool,
    stationary_frames: int,
) -> dict[str, Any]:
    action_fields = [field.name for field in descriptor.resolved_action_fields()]
    return analyze_action_sequence(
        adapter.iter_action_values(episode, action_fields),
        action_fields,
        episode.frame_count,
        trim_stationary_start,
        remove_stationary_segments,
        stationary_frames,
    )


def scan_dataset_motion(
    adapter: Any,
    descriptor: DatasetDescriptor,
    trim_stationary_start: bool,
    remove_stationary_segments: bool,
    stationary_frames: int,
) -> dict[str, Any]:
    action_fields = [field.name for field in descriptor.resolved_action_fields()]
    if not action_fields:
        raise ValueError("The raw dataset adapter did not declare any action fields")

    totals = {
        "source_frames": 0,
        "kept_frames": 0,
        "removed_frames": 0,
        "segments": 0,
        "leading_segments": 0,
        "stationary_segments": 0,
        "episodes_affected": 0,
    }
    for episode in descriptor.episodes:
        result = analyze_episode_motion(
            adapter,
            descriptor,
            episode,
            trim_stationary_start,
            remove_stationary_segments,
            stationary_frames,
        )
        totals["source_frames"] += int(result["source_frames"])
        totals["kept_frames"] += int(result["kept_frames"])
        totals["removed_frames"] += int(result["removed_frames"])
        totals["segments"] += len(result["segments"])
        totals["leading_segments"] += int(result["leading_segments"])
        totals["stationary_segments"] += int(result["stationary_segments"])
        totals["episodes_affected"] += int(result["removed_frames"] > 0)

    return {
        **totals,
        "action_fields": action_fields,
        "episodes": len(descriptor.episodes),
        "fps": descriptor.fps,
        "removed_seconds": totals["removed_frames"] / descriptor.fps,
    }
