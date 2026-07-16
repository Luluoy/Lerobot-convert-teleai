from __future__ import annotations

from pathlib import Path
import json
import tempfile
import time
import unittest

from lerobot_dataconvert.conversion import atomic_write_json
from lerobot_dataconvert.manager import JobManager, JobStore

from .test_core import create_synthetic_hdf5


def wait_for(manager: JobManager, job_id: str, states: set[str], timeout: float = 90) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.store.get(job_id)
        if record and record["state"] in states:
            return record
        time.sleep(0.1)
    raise TimeoutError(f"Job did not reach {states}: {manager.store.get(job_id)}")


class RecoveryTest(unittest.TestCase):
    def test_interrupted_cache_is_recovered_automatically(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lerobot-recovery-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            output = root / "result"
            state_dir = root / "state"
            create_synthetic_hdf5(source, episodes=4, frames=24)
            payload = {
                "adapter": "hdf5_joint",
                "source_path": str(source),
                "output_path": str(output),
                "revision": "v2.1",
                "repo_id": "recovery-test",
                "robot_type": "test_arm",
                "task_instruction": "Move the test arm through the recorded trajectory.",
                "fps": 20,
                "cpu_cores": 2,
                "memory_gb": 4,
                "segment_size": 1,
                "camera_names": {"camera_0": "head", "camera_1": "wrist"},
                "state_names": [f"joint_{index}" for index in range(4)],
                "action_names": [f"joint_{index}" for index in range(4)],
                "adapter_options": {"fps": 20},
                "skip_zero_state": False,
            }

            manager = JobManager(state_dir)
            job = manager.create_job(payload)
            wait_for(manager, job["id"], {"running", "merging", "completed"})
            if manager.store.get(job["id"])["state"] != "completed":
                manager.stop_job(job["id"])
                wait_for(manager, job["id"], {"paused"})
            manager.shutdown()

            record = JobStore(state_dir / "jobs.sqlite3").get(job["id"])
            self.assertIsNotNone(record)
            cache_dir = Path(record["cache_dir"])
            manifest_path = cache_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["state"] != "completed":
                unfinished = next(item for item in manifest["segments"] if item["status"] != "done")
                unfinished["status"] = "running"
                manifest["state"] = "running"
                partial = cache_dir / "segments" / f"segment-{unfinished['id']}"
                partial.mkdir(parents=True, exist_ok=True)
                (partial / "partial.tmp").write_text("incomplete")
                atomic_write_json(manifest_path, manifest)
                store = JobStore(state_dir / "jobs.sqlite3")
                store.patch(job["id"], state="running", phase="convert")

                recovered = JobManager(state_dir)
                final = wait_for(recovered, job["id"], {"completed", "failed"}, timeout=120)
                recovered.shutdown()
                self.assertEqual(final["state"], "completed", final.get("error"))
                self.assertFalse((partial / "partial.tmp").exists())

            self.assertTrue((output / "meta/info.json").exists())

    def test_resume_uses_only_the_output_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lerobot-path-recovery-test-") as temporary:
            root = Path(temporary)
            source = root / "raw"
            output = root / "result"
            create_synthetic_hdf5(source, episodes=2, frames=12)
            payload = {
                "adapter": "hdf5_joint",
                "source_path": str(source),
                "output_path": str(output),
                "revision": "v2.1",
                "repo_id": "path-recovery-test",
                "robot_type": "test_arm",
                "task_instruction": "Move the test arm through the recorded trajectory.",
                "fps": 20,
                "cpu_cores": 2,
                "memory_gb": 4,
                "segment_size": 1,
                "camera_names": {"camera_0": "head", "camera_1": "wrist"},
                "adapter_options": {"fps": 20},
                "skip_zero_state": False,
            }

            original = JobManager(root / "original-state")
            job = original.create_job(payload)
            original.stop_job(job["id"])
            wait_for(original, job["id"], {"paused", "completed"})
            original.shutdown()

            fresh = JobManager(root / "fresh-state")
            recovered = fresh.resume_from_output(str(output))
            final = wait_for(fresh, recovered["id"], {"completed", "failed"}, timeout=120)
            fresh.shutdown()
            self.assertEqual(final["state"], "completed", final.get("error"))
            self.assertTrue((output / "meta/info.json").exists())


if __name__ == "__main__":
    unittest.main()
