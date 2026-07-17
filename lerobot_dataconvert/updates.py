from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import threading
import time
from typing import Any

from .conversion import atomic_write_json


class GitCommandError(RuntimeError):
    pass


class RepositoryUpdater:
    def __init__(self, repository_root: Path, state_dir: Path):
        self.repository_root = repository_root.expanduser().resolve()
        self.state_path = state_dir.expanduser().resolve() / "git-update-state.json"
        self._lock = threading.Lock()

    def check(self, manual: bool = False) -> dict[str, Any]:
        with self._lock:
            saved = self._read_state()
            if saved.get("paused") and not manual:
                return saved
            return self._check_locked(fetch=True)

    def pull(self) -> dict[str, Any]:
        with self._lock:
            saved = self._read_state()
            if saved.get("paused"):
                return saved

            status = self._check_locked(fetch=True)
            if status["status"] != "update_available":
                return status
            try:
                self._git("pull", "--ff-only", "--quiet", timeout=120)
            except (GitCommandError, subprocess.TimeoutExpired) as exc:
                return self._save_error(f"拉取更新失败：{self._error_text(exc)}")

            result = self._check_locked(fetch=False)
            if result["status"] != "up_to_date":
                return result
            result.update(
                {
                    "status": "updated",
                    "message": "已拉取到最新版。请在项目目录运行 ./apply-update.sh，或让 Agent 按 INSTALL.md 完成部署。",
                    "restart_required": True,
                    "updated_at": time.time(),
                }
            )
            return self._save(result)

    def _check_locked(self, fetch: bool) -> dict[str, Any]:
        checked_at = time.time()
        root = self._try_git("rev-parse", "--show-toplevel")
        if root is None or Path(root).resolve() != self.repository_root:
            return self._save(
                {
                    "status": "unavailable",
                    "paused": False,
                    "local_changes": False,
                    "checked_at": checked_at,
                    "message": "当前安装目录不是可更新的 Git 仓库，请询问 Agent。",
                }
            )

        branch = self._try_git("symbolic-ref", "--short", "HEAD")
        if not branch:
            return self._save(
                {
                    "status": "unavailable",
                    "paused": False,
                    "local_changes": False,
                    "checked_at": checked_at,
                    "message": "当前 Git 未处于分支上，请询问 Agent。",
                }
            )

        try:
            changes = [
                line
                for line in self._git("status", "--porcelain=v1", "--untracked-files=normal").splitlines()
                if line.strip()
            ]
        except (GitCommandError, subprocess.TimeoutExpired) as exc:
            return self._save_error(f"检查本地修改失败：{self._error_text(exc)}", branch)
        if changes:
            return self._save(
                {
                    "status": "local_changes",
                    "paused": True,
                    "local_changes": True,
                    "local_change_count": len(changes),
                    "branch": branch,
                    "checked_at": checked_at,
                    "message": "检测到本地修改，自动更新已暂停。请寻求技术帮助或询问 Agent；处理完成后手动点击“检查远端更新”。",
                }
            )

        upstream = self._try_git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        )
        if not upstream:
            return self._save(
                {
                    "status": "unavailable",
                    "paused": False,
                    "local_changes": False,
                    "branch": branch,
                    "checked_at": checked_at,
                    "message": "当前分支没有配置远端跟踪，请询问 Agent。",
                }
            )

        try:
            if fetch:
                self._git("fetch", "--quiet", timeout=60)
            counts = self._git("rev-list", "--left-right", "--count", "HEAD...@{upstream}")
            ahead, behind = (int(value) for value in counts.split())
            head = self._git("rev-parse", "--short=12", "HEAD")
            remote_head = self._git("rev-parse", "--short=12", "@{upstream}")
        except (GitCommandError, subprocess.TimeoutExpired, ValueError) as exc:
            return self._save_error(f"检查远端更新失败：{self._error_text(exc)}", branch, upstream)

        if ahead and behind:
            status = "diverged"
            message = "本地与远端历史已分叉，无法一键更新，请寻求技术帮助或询问 Agent。"
        elif behind:
            status = "update_available"
            message = f"远端有 {behind} 个新提交，可以拉取更新。"
        elif ahead:
            status = "ahead"
            message = f"本地领先远端 {ahead} 个提交，当前没有可拉取更新。"
        else:
            status = "up_to_date"
            message = "当前代码已是远端最新版。"
        return self._save(
            {
                "status": status,
                "paused": False,
                "local_changes": False,
                "branch": branch,
                "upstream": upstream,
                "head": head,
                "remote_head": remote_head,
                "ahead": ahead,
                "behind": behind,
                "checked_at": checked_at,
                "message": message,
            }
        )

    def _save_error(
        self, message: str, branch: str | None = None, upstream: str | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "error",
            "paused": False,
            "local_changes": False,
            "checked_at": time.time(),
            "message": message,
        }
        if branch:
            result["branch"] = branch
        if upstream:
            result["upstream"] = upstream
        return self._save(result)

    def _save(self, value: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(self.state_path, value)
        return dict(value)

    def _read_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return {
            "status": "not_checked",
            "paused": False,
            "local_changes": False,
            "message": "尚未检查远端更新。",
        }

    def _git(self, *args: str, timeout: int = 15) -> str:
        result = self._run_git(*args, timeout=timeout)
        if result.returncode:
            detail = result.stderr.strip().splitlines()
            raise GitCommandError((detail[-1] if detail else "Git command failed")[:500])
        return result.stdout.strip()

    def _try_git(self, *args: str) -> str | None:
        try:
            result = self._run_git(*args, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def _run_git(self, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update({"GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
        return subprocess.run(
            ["git", "-C", str(self.repository_root), *args],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        if isinstance(exc, subprocess.TimeoutExpired):
            return "Git 操作超时"
        text = str(exc) or type(exc).__name__
        return re.sub(r"(https?://)[^/@\s]+@", r"\1", text)[:500]
