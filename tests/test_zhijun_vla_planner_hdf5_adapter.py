from __future__ import annotations

from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

from lerobot_dataconvert.adapters import adapter_catalog, create_adapter
from lerobot_dataconvert.conversion import preview_output_frame, run_segment_worker
from lerobot_dataconvert.models import JobConfig


ADAPTER = "zhijun-vla-planner-HDF5"


def create_zhijun_dataset(root: Path, episodes: int = 2, frames: int = 3) -> None:
    encoded_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    for episode_index in range(episodes):
        episode = root / str(100 + episode_index)
        episode.mkdir(parents=True)
        for frame_index in range(frames):
            path = episode / f"{frame_index}.hdf5"
            with h5py.File(path, "w") as handle:
                joint = np.arange(4, dtype=np.float32) + episode_index + frame_index / 10
                handle.create_dataset("puppet/joint_position", data=joint)
                handle.create_dataset("master/joint_position", data=joint + 0.25)
                handle.create_dataset(
                    "puppet/eef", data=np.arange(8, dtype=np.float32) + frame_index
                )
                handle.create_dataset(
                    "master/eef", data=np.arange(8, dtype=np.float32) + frame_index + 0.5
                )
                handle.create_dataset("puppet/joint_effort", data=joint * 0.1)
                handle.create_dataset(
                    "puppet/6f", data=np.arange(12, dtype=np.float32) + frame_index
                )
                handle.create_dataset(
                    "timestamps", data=np.asarray([frame_index * 50], dtype=np.int32)
                )

                for camera_index in range(4):
                    image = np.zeros((24, 32, 3), dtype=np.uint8)
                    image[:, :, camera_index % 3] = 40 + camera_index * 30
                    image[3 + frame_index : 9 + frame_index, 5:15] = (180, 210, 35)
                    ok, encoded = cv2.imencode(
                        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    )
                    assert ok
                    dataset = handle.create_dataset(
                        f"observations/rgb_images/camera_{camera_index}",
                        (1,),
                        dtype=encoded_dtype,
                    )
                    dataset[0] = encoded.reshape(-1)

                depth = np.full((24, 32), 800 + frame_index, dtype=np.uint16)
                ok, encoded_depth = cv2.imencode(".png", depth)
                assert ok
                depth_dataset = handle.create_dataset(
                    "observations/depth_images/camera_1", (1,), dtype=encoded_dtype
                )
                depth_dataset[0] = encoded_depth.reshape(-1)


def make_config(source: Path, output: Path) -> JobConfig:
    return JobConfig(
        adapter=ADAPTER,
        source_path=str(source),
        output_path=str(output),
        revision="v2.1",
        repo_id="zhijun-synthetic",
        robot_type="rm75",
        task_instruction="Move both arms through the recorded trajectory.",
        fps=20,
        cpu_cores=1,
        memory_gb=2,
        segment_size=1,
        camera_names={"camera_3": "fourth"},
        state_names=[],
        action_names=[],
        field_mapping=[
            {"source": "puppet/joint_position", "target": "observation.state"},
            {"source": "master/joint_position", "target": "action"},
            {"source": "master/eef", "target": "action.eef"},
            {"source": "puppet/eef", "target": "observation.eef"},
            {"source": "puppet/joint_effort", "target": "observation.joint_effort"},
            {"source": "puppet/6f", "target": "observation.force_torque"},
            {"source": "camera_3", "target": "observation.images.fourth"},
        ],
        adapter_options={"fps": 20},
        skip_zero_state=False,
    )


class ZhijunVLAPlannerHDF5AdapterTest(unittest.TestCase):
    def test_catalog_inspect_episode_boundary_fields_and_preview(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zhijun-hdf5-test-") as temporary:
            source = Path(temporary) / "raw"
            create_zhijun_dataset(source)

            catalog = {item["slug"]: item for item in adapter_catalog()}
            self.assertEqual(catalog[ADAPTER]["name"], ADAPTER)

            adapter = create_adapter(ADAPTER, str(source), {"fps": 20})
            descriptor = adapter.inspect()
            self.assertEqual([episode.key for episode in descriptor.episodes], ["100", "101"])
            self.assertEqual(descriptor.total_frames, 6)
            self.assertEqual(
                descriptor.cameras,
                ["camera_0", "camera_1", "camera_2", "camera_3"],
            )
            self.assertTrue(any("Depth images are not exposed" in item for item in descriptor.warnings))

            fields = {field.name: field for field in descriptor.fields}
            self.assertTrue(fields["puppet/joint_position"].is_state)
            self.assertTrue(fields["puppet/eef"].is_state)
            self.assertTrue(fields["puppet/joint_effort"].is_state)
            self.assertTrue(fields["puppet/6f"].is_state)
            self.assertTrue(fields["master/joint_position"].is_action)
            self.assertTrue(fields["master/eef"].is_action)
            self.assertEqual(fields["camera_3"].default_target, "observation.images.fourth")

            single_episode = create_adapter(
                ADAPTER, str(source / "100"), {"fps": 20}
            ).inspect()
            self.assertEqual(len(single_episode.episodes), 1)
            self.assertEqual(single_episode.episodes[0].frame_count, 3)

            sample = next(adapter.iter_frames(descriptor.episodes[0]))
            self.assertEqual(sample.fields["master/eef"].shape, (8,))
            self.assertEqual(sample.images["camera_3"].shape, (24, 32, 3))
            self.assertEqual(
                adapter.preview(descriptor.episodes[0], "camera_3", 99).shape,
                (24, 32, 3),
            )

    def test_action_scan_does_not_decode_images(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zhijun-action-test-") as temporary:
            source = Path(temporary) / "raw"
            create_zhijun_dataset(source, episodes=1, frames=2)
            adapter = create_adapter(ADAPTER, str(source), {"fps": 20})
            episode = adapter.inspect().episodes[0]
            with patch(
                "lerobot_dataconvert.adapters.decode_image",
                side_effect=AssertionError("action scan decoded an image"),
            ):
                actions = list(
                    adapter.iter_action_values(
                        episode, ["master/joint_position", "master/eef"]
                    )
                )
            self.assertEqual(len(actions), 2)
            self.assertEqual(set(actions[0]), {"master/joint_position", "master/eef"})

    def test_segment_conversion_preserves_extended_fields_and_fourth_camera(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zhijun-convert-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            create_zhijun_dataset(source, episodes=1, frames=2)
            adapter = create_adapter(ADAPTER, str(source), {"fps": 20})
            descriptor = adapter.inspect()
            output = root / "segment"
            messages: queue.Queue = queue.Queue()
            config = make_config(source, root / "output")

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
            self.assertIn("action.eef", table.column_names)
            self.assertIn("observation.eef", table.column_names)
            self.assertIn("observation.joint_effort", table.column_names)
            self.assertIn("observation.force_torque", table.column_names)
            self.assertEqual(
                preview_output_frame(
                    output, "v2.1", "observation.images.fourth", 0, 1
                ).shape,
                (24, 32, 3),
            )


if __name__ == "__main__":
    unittest.main()
