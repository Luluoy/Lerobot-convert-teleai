from __future__ import annotations

from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from lerobot_dataconvert.updates import RepositoryUpdater


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


class RepositoryUpdaterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="lerobot-update-test-")
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote.git"
        self.seed = self.root / "seed"
        self.local = self.root / "local"
        subprocess.run(
            ["git", "init", "--bare", "--initial-branch=main", str(self.remote)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "init", "--initial-branch=main", str(self.seed)],
            check=True,
            capture_output=True,
        )
        git(self.seed, "config", "user.name", "Test User")
        git(self.seed, "config", "user.email", "test@example.com")
        (self.seed / "version.txt").write_text("v1\n", encoding="utf-8")
        git(self.seed, "add", "version.txt")
        git(self.seed, "commit", "-m", "initial")
        git(self.seed, "remote", "add", "origin", str(self.remote))
        git(self.seed, "push", "-u", "origin", "main")
        subprocess.run(
            ["git", "clone", "--branch", "main", str(self.remote), str(self.local)],
            check=True,
            capture_output=True,
        )
        self.updater = RepositoryUpdater(self.local, self.root / "state")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_local_changes_pause_automatic_updates_until_manual_check(self) -> None:
        self.assertEqual(self.updater.check()["status"], "up_to_date")
        local_only = self.local / "local-only.txt"
        local_only.write_text("do not overwrite\n", encoding="utf-8")

        dirty = self.updater.check()
        self.assertEqual(dirty["status"], "local_changes")
        self.assertTrue(dirty["paused"])
        self.assertEqual(dirty["local_change_count"], 1)
        recorded = json.loads(self.updater.state_path.read_text(encoding="utf-8"))
        self.assertTrue(recorded["local_changes"])

        local_only.unlink()
        restarted = RepositoryUpdater(self.local, self.root / "state")
        self.assertEqual(restarted.check()["status"], "local_changes")
        self.assertEqual(restarted.pull()["status"], "local_changes")
        resumed = restarted.check(manual=True)
        self.assertEqual(resumed["status"], "up_to_date")
        self.assertFalse(resumed["paused"])

    def test_remote_update_can_be_pulled_with_fast_forward_only(self) -> None:
        (self.seed / "version.txt").write_text("v2\n", encoding="utf-8")
        git(self.seed, "add", "version.txt")
        git(self.seed, "commit", "-m", "update")
        git(self.seed, "push")

        available = self.updater.check()
        self.assertEqual(available["status"], "update_available")
        self.assertEqual(available["behind"], 1)
        updated = self.updater.pull()
        self.assertEqual(updated["status"], "updated")
        self.assertTrue(updated["restart_required"])
        self.assertEqual((self.local / "version.txt").read_text(encoding="utf-8"), "v2\n")
        self.assertEqual(self.updater.check()["status"], "up_to_date")


if __name__ == "__main__":
    unittest.main()
