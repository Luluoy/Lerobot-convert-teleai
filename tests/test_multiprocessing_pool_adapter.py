from __future__ import annotations

from pathlib import Path
import json
import pickle
import queue
import tempfile
import unittest

import cv2
import numpy as np
import pyarrow.parquet as pq

from lerobot_dataconvert.adapters import create_adapter
from lerobot_dataconvert.conversion import preview_output_frame, run_segment_worker
from lerobot_dataconvert.manager import _normalize_field_mapping
from lerobot_dataconvert.models import JobConfig


def create_pool_dataset(root: Path, episodes: int = 2, frames: int = 3) -> None:
    root.mkdir(parents=True)
    joint_names = tuple(f"joint_{index}" for index in range(4))
    eef_names = tuple(f"eef_{index}" for index in range(16))
    cameras = {
        "Cam1": {"camera_name": "head", "shape": (32, 48)},
        "Cam2": {"camera_name": "left_wrist", "shape": (24, 36)},
    }
    for episode_index in range(episodes):
        episode = root / f"episode_{episode_index:06d}"
        for directory in ("eef_action", "joint_action", "joint_state", *cameras):
            (episode / directory).mkdir(parents=True)

        for frame_index in range(frames):
            timestamp_ns = 1_700_000_000_000_000_000 + episode_index * 1_000_000_000 + frame_index * 50_000_000
            monotonic_ns = 5_000_000_000 + frame_index * 50_000_000
            stem = f"frame_{frame_index:09d}_{timestamp_ns}"
            qpos = np.arange(4, dtype=np.float32) + 1 + episode_index + frame_index / 10
            common = {
                "schema_version": 3,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "monotonic_timestamp_ns": monotonic_ns,
            }
            camera_records: dict[str, dict] = {}
            for camera_index, (name, camera) in enumerate(cameras.items()):
                height, width = camera["shape"]
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                rgb[:, :, camera_index] = 80 + frame_index * 20
                rgb[3:10, 4:16] = (190, 220, 35)
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                assert ok
                camera_timestamp_ns = timestamp_ns + camera_index * 1_000_000
                image_path = f"{name}/frame_{frame_index:09d}_{camera_timestamp_ns}.png"
                (episode / image_path).write_bytes(encoded.tobytes())
                camera_records[name] = {
                    "timestamp_ns": camera_timestamp_ns,
                    "image_path": image_path,
                    "metadata": {
                        "camera": camera["camera_name"],
                        "dataset_name": name,
                        "serial": f"SN-{camera_index}",
                        "frame_seq": frame_index,
                        "capture_monotonic_ns": monotonic_ns + camera_index * 1_000_000,
                    },
                }

            records = {
                "eef_action": {**common, "action": np.linspace(0.0, 1.0, 16, dtype=np.float32)},
                "joint_action": {**common, "action": qpos.copy()},
                "joint_state": {
                    **common,
                    "qpos": qpos.copy(),
                    "qvel": qpos * 0.1,
                    "torque": qpos * 0.01,
                    "robot_timing": {"sample_monotonic_ns": monotonic_ns},
                    "robots": {},
                    "cameras": camera_records,
                },
            }
            for stream, record in records.items():
                (episode / stream / f"{stem}.pkl").write_bytes(
                    pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
                )

        streams: dict[str, dict] = {
            "eef_action": {
                "type": "action",
                "path": "eef_action",
                "nominal_fps": 20,
                "order": list(eef_names),
            },
            "joint_action": {
                "type": "action",
                "path": "joint_action",
                "nominal_fps": 20,
                "order": list(joint_names),
            },
            "joint_state": {
                "type": "state",
                "path": "joint_state",
                "nominal_fps": 20,
                "order": list(joint_names),
                "fields": {name: list(joint_names) for name in ("qpos", "qvel", "torque")},
            },
        }
        for camera_index, (name, camera) in enumerate(cameras.items()):
            height, width = camera["shape"]
            streams[name] = {
                "type": "camera",
                "path": name,
                "nominal_fps": 20,
                "camera_name": camera["camera_name"],
                "serial": f"SN-{camera_index}",
                "width": width,
                "height": height,
                "encoding": "png",
            }
        meta = {
            "schema": "teleaxis_collector_episode",
            "schema_version": 3,
            "episode_index": episode_index,
            "status": "complete",
            "reason": "operator_finish",
            "frame_count": frames,
            "saved_frame_count": frames,
            "streams": streams,
            "save_errors": [],
            "validation": {"status": "PASS"},
        }
        (episode / "META").mkdir()
        (episode / "META" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    incomplete = root / "episode_999999" / "META"
    incomplete.mkdir(parents=True)
    incomplete_meta = dict(meta)
    incomplete_meta.update(
        {"episode_index": 999999, "status": "recording", "frame_count": 0, "saved_frame_count": 0}
    )
    (incomplete / "meta.json").write_text(json.dumps(incomplete_meta), encoding="utf-8")


def pool_config(source: Path, output: Path) -> JobConfig:
    return JobConfig(
        adapter="multiprocessing_pool_dataset",
        source_path=str(source),
        output_path=str(output),
        revision="v2.1",
        repo_id="pool-synthetic",
        robot_type="acone",
        task_instruction="Move both arms through the recorded trajectory.",
        fps=20,
        cpu_cores=1,
        memory_gb=2,
        segment_size=1,
        camera_names={"Cam1": "head", "Cam2": "left_wrist"},
        state_names=[],
        action_names=[],
        field_mapping={
            "joint_state/qpos": "observation.state",
            "joint_state/qvel": "observation.velocity",
            "joint_state/torque": "observation.effort",
            "joint_action/action": "action",
            "eef_action/action": "observation.eef_pose",
            "Cam1": "observation.images.head",
            "Cam2": "observation.images.left_wrist",
        },
        adapter_options={},
        skip_zero_state=False,
    )


class MultiProcessingPoolDatasetTest(unittest.TestCase):
    def test_schema_v3_fields_preview_and_conversion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-dataset-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            create_pool_dataset(source)
            adapter = create_adapter("multiprocessing_pool_dataset", str(source))
            descriptor = adapter.inspect()

            self.assertEqual(len(descriptor.episodes), 2)
            self.assertEqual(descriptor.total_frames, 6)
            self.assertEqual(descriptor.fps, 20)
            self.assertEqual(descriptor.cameras, ["Cam1", "Cam2"])
            self.assertEqual(descriptor.camera_shapes, {"Cam1": (32, 48, 3), "Cam2": (24, 36, 3)})
            fields = {field.name: field for field in descriptor.fields}
            self.assertEqual(fields["joint_state/qpos"].default_target, "observation.state")
            self.assertEqual(fields["joint_action/action"].default_target, "action")
            self.assertEqual(fields["Cam1"].default_target, "observation.images.head")
            self.assertTrue(fields["Cam1"].is_image)
            self.assertEqual(fields["eef_action/action"].shape, (16,))
            self.assertIn("episode_999999: skipped", descriptor.warnings[0])

            preview = adapter.preview(descriptor.episodes[0], "Cam1", 1)
            self.assertEqual(preview.shape, (32, 48, 3))
            samples = list(adapter.iter_frames(descriptor.episodes[0]))
            self.assertEqual(len(samples), 3)
            self.assertAlmostEqual(samples[1].timestamp or 0.0, 0.05)
            np.testing.assert_allclose(samples[2].fields["joint_state/qvel"], samples[2].state * 0.1)

            output = root / "segment"
            messages: queue.Queue = queue.Queue()
            config = pool_config(source, output)
            run_segment_worker(
                {
                    "segment_id": "000000",
                    "output_dir": str(output),
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
            self.assertTrue((output / ".segment-complete.json").is_file())
            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            self.assertIn("observation.velocity", table.column_names)
            self.assertIn("observation.effort", table.column_names)
            self.assertIn("observation.eef_pose", table.column_names)
            self.assertEqual(
                preview_output_frame(output, "v2.1", "observation.images.head", 0, 1).shape,
                (32, 48, 3),
            )

    def test_camera_reference_cannot_escape_episode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-path-test-") as temporary:
            source = Path(temporary) / "raw"
            create_pool_dataset(source, episodes=1, frames=1)
            adapter = create_adapter("multiprocessing_pool_dataset", str(source))
            descriptor = adapter.inspect()
            state_path = next((source / "episode_000000" / "joint_state").glob("*.pkl"))
            record = pickle.loads(state_path.read_bytes())
            record["cameras"]["Cam1"]["image_path"] = "../outside.png"
            state_path.write_bytes(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))

            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                adapter.preview(descriptor.episodes[0], "Cam1", 0)

    def test_invalid_field_mappings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-mapping-test-") as temporary:
            source = Path(temporary) / "raw"
            create_pool_dataset(source, episodes=1, frames=1)
            descriptor = create_adapter("multiprocessing_pool_dataset", str(source)).inspect()
            camera_names = {"Cam1": "head", "Cam2": "left_wrist"}

            with self.assertRaisesRegex(ValueError, "must be unique"):
                _normalize_field_mapping(
                    {
                        "field_mapping": {
                            "joint_state/qpos": "action",
                            "joint_action/action": "action",
                        }
                    },
                    descriptor,
                    camera_names,
                )
            with self.assertRaisesRegex(ValueError, "Image field"):
                _normalize_field_mapping(
                    {"field_mapping": {"Cam1": "observation.state"}},
                    descriptor,
                    camera_names,
                )


if __name__ == "__main__":
    unittest.main()
