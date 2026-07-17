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
from lerobot_dataconvert.motion import scan_dataset_motion


def create_pool_dataset(
    root: Path,
    episodes: int = 2,
    frames: int = 3,
    action_pattern: list[float] | None = None,
    sample_fps: int = 20,
) -> None:
    if action_pattern is not None and len(action_pattern) != frames:
        raise ValueError("action_pattern must contain one value per frame")
    root.mkdir(parents=True)
    joint_names = tuple(f"joint_{index}" for index in range(4))
    eef_names = tuple(f"eef_{index}" for index in range(16))
    cameras = {
        "Cam1": {"camera_name": "head", "shape": (32, 48)},
        "Cam2": {"camera_name": "left_wrist", "shape": (24, 36)},
    }
    step_ns = round(1_000_000_000 / sample_fps)
    for episode_index in range(episodes):
        episode = root / f"episode_{episode_index:06d}"
        for directory in ("eef_action", "joint_action", "joint_state", *cameras):
            (episode / directory).mkdir(parents=True)

        for frame_index in range(frames):
            timestamp_ns = (
                1_700_000_000_000_000_000
                + episode_index * 2_000_000_000
                + frame_index * step_ns
            )
            monotonic_ns = 5_000_000_000 + frame_index * step_ns
            stem = f"frame_{frame_index:09d}_{timestamp_ns}"
            action_offset = action_pattern[frame_index] if action_pattern is not None else frame_index / 10
            qpos = np.arange(4, dtype=np.float32) + 1 + episode_index + action_offset
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
                camera_timestamp_ns = timestamp_ns
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
                "nominal_fps": sample_fps,
                "order": list(eef_names),
            },
            "joint_action": {
                "type": "action",
                "path": "joint_action",
                "nominal_fps": sample_fps,
                "order": list(joint_names),
            },
            "joint_state": {
                "type": "state",
                "path": "joint_state",
                "nominal_fps": sample_fps,
                "order": list(joint_names),
                "fields": {name: list(joint_names) for name in ("qpos", "qvel", "torque")},
            },
        }
        for camera_index, (name, camera) in enumerate(cameras.items()):
            height, width = camera["shape"]
            streams[name] = {
                "type": "camera",
                "path": name,
                "nominal_fps": sample_fps,
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
        field_mapping=[
            {"source": "joint_state/qpos", "target": "observation.state"},
            {"source": "joint_state/qpos", "target": "observation.qpos_copy"},
            {"source": "joint_state/qvel", "target": "observation.velocity"},
            {"source": "joint_state/torque", "target": "observation.effort"},
            {"source": "joint_action/action", "target": "action"},
            {"source": "eef_action/action", "target": "observation.eef_pose"},
            {"source": "Cam1", "target": "observation.images.head"},
            {"source": "Cam2", "target": "observation.images.left_wrist"},
        ],
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
            self.assertEqual(descriptor.max_output_fps, 20)
            self.assertEqual(descriptor.cameras, ["Cam1", "Cam2"])
            self.assertEqual(descriptor.camera_shapes, {"Cam1": (32, 48, 3), "Cam2": (24, 36, 3)})
            fields = {field.name: field for field in descriptor.fields}
            self.assertEqual(fields["joint_state/qpos"].default_target, "observation.state")
            self.assertTrue(fields["joint_state/qpos"].is_state)
            self.assertTrue(fields["joint_state/qvel"].is_state)
            self.assertTrue(fields["joint_state/torque"].is_state)
            self.assertEqual(fields["joint_action/action"].default_target, "action")
            self.assertTrue(fields["joint_action/action"].is_action)
            self.assertTrue(fields["eef_action/action"].is_action)
            self.assertEqual(fields["Cam1"].default_target, "observation.images.head")
            self.assertTrue(fields["Cam1"].is_image)
            self.assertEqual(fields["eef_action/action"].shape, (16,))
            self.assertTrue(all(field.fps == 20 for field in fields.values()))
            self.assertEqual(descriptor.to_dict()["max_output_fps"], 20)
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
            self.assertIn("observation.qpos_copy", table.column_names)
            self.assertIn("observation.velocity", table.column_names)
            self.assertIn("observation.effort", table.column_names)
            self.assertIn("observation.eef_pose", table.column_names)
            self.assertEqual(
                preview_output_frame(output, "v2.1", "observation.images.head", 0, 1).shape,
                (32, 48, 3),
            )

    def test_zero_state_action_fields_are_forward_filled_independently(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-zero-fill-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            create_pool_dataset(source, episodes=1, frames=4)
            episode = source / "episode_000000"

            def replace_field(stream: str, frame_index: int, name: str, value: np.ndarray) -> None:
                path = next((episode / stream).glob(f"frame_{frame_index:09d}_*.pkl"))
                record = pickle.loads(path.read_bytes())
                record[name] = np.asarray(value, dtype=np.float32)
                path.write_bytes(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))

            zeros4 = np.zeros(4, dtype=np.float32)
            replace_field("joint_state", 0, "torque", zeros4)
            replace_field("joint_state", 1, "qpos", zeros4)
            replace_field("joint_state", 1, "qvel", np.array([0, 0, 0, 1]))
            replace_field("joint_action", 1, "action", zeros4)
            replace_field("joint_state", 2, "qpos", zeros4)
            replace_field("joint_state", 2, "qvel", zeros4)
            replace_field("eef_action", 2, "action", np.zeros(16, dtype=np.float32))
            replace_field("joint_state", 3, "torque", zeros4)

            adapter = create_adapter("multiprocessing_pool_dataset", str(source))
            descriptor = adapter.inspect()
            config = pool_config(source, root / "output")
            config.fill_zero_state_action = True
            output = root / "segment"
            messages: queue.Queue = queue.Queue()
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

            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")

            def column(name: str) -> np.ndarray:
                return np.asarray(table[name].to_pylist(), dtype=np.float32)

            state = column("observation.state")
            velocity = column("observation.velocity")
            effort = column("observation.effort")
            action = column("action")
            eef_action = column("observation.eef_pose")
            np.testing.assert_array_equal(state[1], state[0])
            np.testing.assert_array_equal(state[2], state[0])
            np.testing.assert_array_equal(column("observation.qpos_copy"), state)
            np.testing.assert_array_equal(velocity[1], [0, 0, 0, 1])
            np.testing.assert_array_equal(velocity[2], velocity[1])
            np.testing.assert_array_equal(action[1], action[0])
            np.testing.assert_array_equal(eef_action[2], eef_action[1])
            np.testing.assert_array_equal(effort[0], zeros4)
            np.testing.assert_array_equal(effort[3], effort[2])

    def test_async_streams_are_aligned_by_nearest_timestamp(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-fps-test-") as temporary:
            source = Path(temporary) / "raw"
            create_pool_dataset(source, episodes=1, frames=9, sample_fps=8)
            episode = source / "episode_000000"

            for stream_name in ("eef_action", "Cam1", "Cam2"):
                for path in (episode / stream_name).iterdir():
                    if int(path.name.split("_")[1]) % 2:
                        path.unlink()
            action_middle = next(
                path
                for path in (episode / "joint_action").iterdir()
                if int(path.name.split("_")[1]) == 4
            )
            action_middle.unlink()

            meta_path = episode / "META" / "meta.json"
            meta = json.loads(meta_path.read_text())
            for stream_name, stream in meta["streams"].items():
                stream["actual_fps"] = 4 if stream_name in {"eef_action", "Cam1", "Cam2"} else 8
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

            adapter = create_adapter("multiprocessing_pool_dataset", str(source), {"fps": 2})
            descriptor = adapter.inspect()
            self.assertEqual(descriptor.fps, 2)
            self.assertEqual(descriptor.max_output_fps, 4)
            self.assertEqual(descriptor.total_frames, 3)
            rates = {field.name: field.fps for field in descriptor.fields}
            self.assertEqual(rates["joint_state/qpos"], 8)
            self.assertEqual(rates["joint_action/action"], 8)
            self.assertEqual(rates["eef_action/action"], 4)
            self.assertEqual(rates["Cam1"], 4)

            samples = list(adapter.iter_frames(descriptor.episodes[0]))
            self.assertEqual([sample.timestamp for sample in samples], [0.0, 0.5, 1.0])
            np.testing.assert_allclose(
                [sample.state[0] for sample in samples], [1.0, 1.4, 1.8]
            )
            np.testing.assert_allclose(
                [sample.action[0] for sample in samples], [1.0, 1.3, 1.8]
            )
            self.assertEqual(int(samples[1].images["Cam1"][0, 0, 0]), 160)
            self.assertEqual(
                int(adapter.preview(descriptor.episodes[0], "Cam1", 1)[0, 0, 0]), 160
            )
            action_values = list(
                adapter.iter_action_values(
                    descriptor.episodes[0], ["joint_action/action", "eef_action/action"]
                )
            )
            self.assertEqual(len(action_values), 3)
            self.assertAlmostEqual(float(action_values[1]["joint_action/action"][0]), 1.3)

            with self.assertRaisesRegex(ValueError, "exceeds the minimum field FPS 4"):
                create_adapter(
                    "multiprocessing_pool_dataset", str(source), {"fps": 5}
                ).inspect()

    def test_motion_scan_matches_filtered_conversion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-motion-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            create_pool_dataset(
                source,
                episodes=1,
                frames=8,
                action_pattern=[0, 0, 0, 1, 1, 1, 1, 2],
            )
            zero_action_path = next(
                (source / "episode_000000" / "joint_action").glob("frame_000000004_*.pkl")
            )
            zero_action = pickle.loads(zero_action_path.read_bytes())
            zero_action["action"] = np.zeros(4, dtype=np.float32)
            zero_action_path.write_bytes(
                pickle.dumps(zero_action, protocol=pickle.HIGHEST_PROTOCOL)
            )
            adapter = create_adapter("multiprocessing_pool_dataset", str(source))
            descriptor = adapter.inspect()
            scan = scan_dataset_motion(adapter, descriptor, True, True, 3, True)
            self.assertEqual(scan["action_fields"], ["joint_action/action", "eef_action/action"])
            self.assertEqual(scan["leading_segments"], 1)
            self.assertEqual(scan["stationary_segments"], 1)
            self.assertEqual(scan["removed_frames"], 6)
            self.assertEqual(scan["kept_frames"], 2)
            self.assertEqual(scan["removed_seconds"], 0.3)

            config = pool_config(source, root / "output")
            config.trim_stationary_start = True
            config.remove_stationary_segments = True
            config.stationary_frames = 3
            config.fill_zero_state_action = True
            messages: queue.Queue = queue.Queue()
            output = root / "segment"
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
            marker = json.loads((output / ".segment-complete.json").read_text())
            self.assertEqual(marker["processed_frames"], 8)
            self.assertEqual(marker["removed_frames"], 6)
            self.assertEqual(marker["removed_segments"], 2)
            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(table.num_rows, 2)

    def test_camera_reference_cannot_escape_episode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-path-test-") as temporary:
            source = Path(temporary) / "raw"
            create_pool_dataset(source, episodes=1, frames=1)
            adapter = create_adapter("multiprocessing_pool_dataset", str(source))
            descriptor = adapter.inspect()
            meta_path = source / "episode_000000" / "META" / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["streams"]["Cam1"]["path"] = "../outside"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                adapter.preview(descriptor.episodes[0], "Cam1", 0)

    def test_invalid_field_mappings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pool-mapping-test-") as temporary:
            source = Path(temporary) / "raw"
            create_pool_dataset(source, episodes=1, frames=1)
            descriptor = create_adapter("multiprocessing_pool_dataset", str(source)).inspect()
            camera_names = {"Cam1": "head", "Cam2": "left_wrist"}

            repeated_source = _normalize_field_mapping(
                {
                    "field_mapping": [
                        {"source": "joint_state/qpos", "target": "observation.state"},
                        {"source": "joint_state/qpos", "target": "observation.qpos_copy"},
                    ]
                },
                descriptor,
                camera_names,
            )
            self.assertEqual(
                [row["source"] for row in repeated_source],
                ["joint_state/qpos", "joint_state/qpos"],
            )
            with self.assertRaisesRegex(ValueError, "must be unique"):
                _normalize_field_mapping(
                    {
                        "field_mapping": [
                            {"source": "joint_state/qpos", "target": "action"},
                            {"source": "joint_action/action", "target": "action"},
                        ]
                    },
                    descriptor,
                    camera_names,
                )
            with self.assertRaisesRegex(ValueError, "Image field"):
                _normalize_field_mapping(
                    {
                        "field_mapping": [
                            {"source": "Cam1", "target": "observation.state"}
                        ]
                    },
                    descriptor,
                    camera_names,
                )


if __name__ == "__main__":
    unittest.main()
