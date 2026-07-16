from __future__ import annotations

from pathlib import Path
import json
import queue
import tempfile
import unittest

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


def make_config(source: Path, output: Path, revision: str = "v2.1") -> JobConfig:
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
        adapter_options={"fps": 20},
        skip_zero_state=False,
    )


class ConversionTest(unittest.TestCase):
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
