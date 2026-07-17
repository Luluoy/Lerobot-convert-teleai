from __future__ import annotations

from pathlib import Path
import json
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

from lerobot_dataconvert.adapters import create_adapter
from lerobot_dataconvert.conversion import (
    convert_v21_to_v30,
    merge_v21_segments,
    preview_output_frame,
    run_segment_worker,
)
from lerobot_dataconvert.models import JobConfig
from lerobot_dataconvert.manager import MAX_CPU_LIMIT_PERCENT, _CpuDutyCycleGovernor
from lerobot_dataconvert.motion import analyze_action_sequence


def create_synthetic_hdf5(root: Path, episodes: int = 3, frames: int = 6) -> None:
    root.mkdir(parents=True)
    encoded_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    for episode_index in range(episodes):
        path = root / f"episode_{episode_index}.hdf5"
        with h5py.File(path, "w") as handle:
            state = np.arange(frames * 4, dtype=np.float32).reshape(frames, 4) + episode_index
            handle.create_dataset("observations/qpos", data=state)
            handle.create_dataset("action", data=state + 0.25)
            for camera_index, camera_path in enumerate(
                ["observations/images/head", "observations/images/left_wrist"]
            ):
                dataset = handle.create_dataset(camera_path, (frames,), dtype=encoded_dtype)
                for frame_index in range(frames):
                    image = np.zeros((64, 96, 3), dtype=np.uint8)
                    image[:, :, camera_index] = 70 + episode_index * 20
                    image[8 + frame_index : 24 + frame_index, 12:46] = (180, 210, 35)
                    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    assert ok
                    dataset[frame_index] = encoded.reshape(-1)


def make_config(
    source: Path, output: Path, revision: str = "v2.1", video_crf: int = 30
) -> JobConfig:
    return JobConfig(
        adapter="hdf5_joint",
        source_path=str(source),
        output_path=str(output),
        revision=revision,
        repo_id="synthetic",
        robot_type="test_arm",
        task_instruction="Move the test arm through the recorded trajectory.",
        fps=20,
        cpu_cores=2,
        memory_gb=4,
        segment_size=1,
        camera_names={"camera_0": "head", "camera_1": "wrist"},
        state_names=[f"joint_{index}" for index in range(4)],
        action_names=[f"joint_{index}" for index in range(4)],
        video_crf=video_crf,
        adapter_options={"fps": 20},
        skip_zero_state=False,
    )


class ConversionTest(unittest.TestCase):
    def test_cpu_governor_caps_worker_duty_cycle_at_95_percent(self) -> None:
        with (
            patch("lerobot_dataconvert.manager._suspend_processes", return_value=[]) as suspend,
            patch("lerobot_dataconvert.manager._resume_processes") as resume,
        ):
            governor = _CpuDutyCycleGovernor(lambda: [object()], 100, period_seconds=0.02)
            self.assertEqual(governor.limit_percent, MAX_CPU_LIMIT_PERCENT)
            self.assertAlmostEqual(governor.run_seconds, 0.019)
            self.assertAlmostEqual(governor.pause_seconds, 0.001)
            with governor:
                time.sleep(0.07)
            self.assertGreaterEqual(suspend.call_count, 2)
            self.assertEqual(resume.call_count, suspend.call_count)

    def test_motion_analysis_uses_all_declared_action_fields(self) -> None:
        values = [
            {"joint": [0], "eef": [0]},
            {"joint": [0], "eef": [0]},
            {"joint": [0], "eef": [1]},
            {"joint": [0], "eef": [1]},
            {"joint": [0], "eef": [1]},
            {"joint": [2], "eef": [1]},
        ]
        result = analyze_action_sequence(values, ["joint", "eef"], 6, True, True, 3)
        self.assertEqual(
            result["segments"],
            [
                {"start": 0, "end": 2, "kind": "leading", "frames": 2},
                {"start": 3, "end": 5, "kind": "stationary", "frames": 2},
            ],
        )
        self.assertEqual(result["kept_frames"], 2)

    def test_motion_analysis_forward_fills_each_zero_action_field(self) -> None:
        values = [
            {"joint": [1], "eef": [2]},
            {"joint": [0], "eef": [2]},
            {"joint": [0], "eef": [0]},
            {"joint": [3], "eef": [2]},
        ]
        without_fill = analyze_action_sequence(values, ["joint", "eef"], 4, False, True, 2)
        with_fill = analyze_action_sequence(
            values, ["joint", "eef"], 4, False, True, 2, True
        )
        self.assertEqual(without_fill["removed_frames"], 0)
        self.assertEqual(with_fill["removed_frames"], 2)
        self.assertEqual(
            with_fill["segments"],
            [{"start": 1, "end": 3, "kind": "stationary", "frames": 2}],
        )

    def test_video_crf_is_forwarded_to_lerobot_encoder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lerobot-video-crf-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            create_synthetic_hdf5(source, episodes=1, frames=3)
            descriptor = create_adapter("hdf5_joint", str(source), {"fps": 20}).inspect()
            config = make_config(source, root / "output", video_crf=22)
            legacy_config = config.to_dict()
            legacy_config.pop("video_crf")
            legacy_config.pop("cpu_limit_percent")
            legacy_config.pop("fill_zero_state_action")
            legacy_config["field_mapping"] = {"state": "observation.state"}
            self.assertEqual(JobConfig.from_dict(legacy_config).video_crf, 30)
            self.assertEqual(JobConfig.from_dict(legacy_config).cpu_limit_percent, 95)
            self.assertFalse(JobConfig.from_dict(legacy_config).fill_zero_state_action)
            self.assertEqual(
                JobConfig.from_dict(legacy_config).field_mapping,
                [{"source": "state", "target": "observation.state"}],
            )

            try:
                from lerobot.datasets import lerobot_dataset as dataset_module
            except ModuleNotFoundError:
                from lerobot.common.datasets import lerobot_dataset as dataset_module

            encoded_crfs: list[int] = []
            original_encoder = dataset_module.encode_video_frames

            def recording_encoder(*args, **kwargs):
                encoded_crfs.append(kwargs["crf"])
                return original_encoder(*args, **kwargs)

            messages: queue.Queue = queue.Queue()
            with patch.object(dataset_module, "encode_video_frames", recording_encoder):
                run_segment_worker(
                    {
                        "segment_id": "000000",
                        "output_dir": str(root / "segment"),
                        "config": config.to_dict(),
                        "descriptor": descriptor.to_dict(),
                        "source_indices": [0],
                        "episodes": [descriptor.episodes[0].to_dict()],
                        "cpu_id": None,
                    },
                    messages,
                )

            results = []
            while not messages.empty():
                results.append(messages.get())
            failures = [message for message in results if message["type"] == "failed"]
            self.assertFalse(failures, failures[0]["traceback"] if failures else "")
            self.assertEqual(encoded_crfs, [22, 22])

    def test_adapter_segment_merge_and_revisions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lerobot-convert-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            create_synthetic_hdf5(source)
            adapter = create_adapter("hdf5_joint", str(source), {"fps": 20})
            descriptor = adapter.inspect()
            self.assertEqual(len(descriptor.episodes), 3)
            self.assertEqual(descriptor.total_frames, 18)
            self.assertEqual(descriptor.state_dim, 4)
            self.assertEqual(adapter.preview(descriptor.episodes[1], "camera_0", 3).shape, (64, 96, 3))

            config = make_config(source, root / "output-v21")
            segment_dirs = []
            for index, episode in enumerate(descriptor.episodes):
                output = root / "segments" / f"segment-{index:06d}"
                messages: queue.Queue = queue.Queue()
                run_segment_worker(
                    {
                        "segment_id": f"{index:06d}",
                        "output_dir": str(output),
                        "config": config.to_dict(),
                        "descriptor": descriptor.to_dict(),
                        "source_indices": [index],
                        "episodes": [episode.to_dict()],
                        "cpu_id": None,
                    },
                    messages,
                )
                results = []
                while not messages.empty():
                    results.append(messages.get())
                failures = [message for message in results if message["type"] == "failed"]
                self.assertFalse(failures, failures[0]["traceback"] if failures else "")
                self.assertTrue((output / ".segment-complete.json").exists())
                segment_dirs.append(output)

            v21 = root / "output-v21"
            result = merge_v21_segments(segment_dirs, v21, config, descriptor)
            self.assertEqual(result["episodes"], 3)
            self.assertEqual(result["frames"], 18)
            info = json.loads((v21 / "meta" / "info.json").read_text())
            self.assertEqual(info["codebase_version"], "v2.1")
            last_table = pq.read_table(v21 / "data/chunk-000/episode_000002.parquet")
            self.assertEqual(last_table["index"][0].as_py(), 12)
            self.assertEqual(last_table["episode_index"][0].as_py(), 2)
            self.assertEqual(preview_output_frame(v21, "v2.1", "observation.images.head", 2, 4).shape, (64, 96, 3))

            v30 = root / "output-v30"
            packed = convert_v21_to_v30(v21, v30)
            self.assertEqual(packed["episodes"], 3)
            v30_info = json.loads((v30 / "meta" / "info.json").read_text())
            self.assertEqual(v30_info["codebase_version"], "v3.0")
            self.assertTrue((v30 / "meta/tasks.parquet").exists())
            self.assertTrue((v30 / "meta/episodes/chunk-000/file-000.parquet").exists())
            self.assertEqual(preview_output_frame(v30, "v3.0", "observation.images.wrist", 1, 2).shape, (64, 96, 3))


if __name__ == "__main__":
    unittest.main()
