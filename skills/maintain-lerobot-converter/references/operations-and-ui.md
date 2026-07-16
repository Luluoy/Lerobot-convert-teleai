# Operations And UI

Last verified: 2026-07-16.

Apply the freshness contract in `../SKILL.md`: update this file in the same change whenever the
implementation makes a statement below inaccurate.

## Runtime And Installation

- Supported Python: 3.10 or newer with dependencies from `pyproject.toml`.
- `start.sh` should prefer the repository `.venv`, retain the local `lerobot21` compatibility
  fallback, then use `python3`; `LEROBOT_DATACONVERT_PYTHON` overrides all defaults.
- Default bind address: `127.0.0.1:8765`.
- Runtime state defaults to `~/.local/share/lerobot-dataconvert` and can be overridden with
  `LEROBOT_DATACONVERT_STATE` or `--state-dir`.
- Package initialization defaults `ARROW_DEFAULT_MEMORY_POOL` to `system`. PyArrow 25's mimalloc
  backend can segfault after repeated v3 preview reads on short-lived HTTP threads; preserve this
  default unless the allocator and repeated-preview E2E have been verified together.
- `install-systemd-service.sh` creates and enables the user unit at
  `~/.config/systemd/user/lerobot-dataconvert.service`.
- The PWA does not start a backend by itself. The user systemd service is the supported automatic
  backend startup mechanism.

Keep `INSTALL.md`, `README.md`, `start.sh`, the systemd installer, and this reference consistent.

## Backend Availability UX

The cached PWA shell can open while the backend is offline. In that state, display a persistent,
full-width backend notice that:

- distinguishes the missing backend from browser network status;
- tells first-time users to ask an Agent to follow root `INSTALL.md`;
- recommends `systemctl --user restart lerobot-dataconvert` when already installed;
- offers a manual recheck and also retries automatically;
- disappears after bootstrap/job polling succeeds.

Do not imply that a browser or installed PWA can execute system commands. Keep core conversion
controls unavailable until bootstrap data exists.

## Frontend Contract

The UI is a no-build PWA using committed static HTML/CSS/JS and vendored Lucide icons. Preserve its
Industrial design system: warm black surfaces, acid-lime semantic signal, monospace typography,
flat 1 px rules, square controls, and tabular numerics. Use Lucide icons with accessible labels for
tool actions. Do not introduce gradients, shadows, rounded cards, fabricated telemetry, or nested
cards.

Keep desktop and 390 px mobile layouts free of horizontal body overflow, text overlap, and control
resizing. The table may own an internal horizontal scroller. When changing cached shell assets,
bump the cache key in `static/sw.js` so installed PWAs receive the update.

The delete task action is available only when a job is not active. It calls metadata-only DELETE,
removes the row/detail selection, and explicitly tells the user that local files were retained.

Resource controls expose both a core count and a CPU utilization ceiling. The percentage may be
lowered but must never exceed 95 in HTML, API normalization, persisted `JobConfig`, or scheduler
enforcement. Displaying a limit without enforcing it on worker/encoder processes is incorrect.

## Verification Matrix

Run focused tests while iterating, then use the applicable final checks:

```bash
python -m compileall -q lerobot_dataconvert tests
python -m unittest -v
node --check lerobot_dataconvert/static/app.js
git diff --check
```

For UI/PWA changes, run a backend and then:

```bash
node tests/ui_check.mjs
```

Inspect its desktop and mobile screenshots, not only the exit code. The E2E check performs real
conversion, offline-shell verification, backend-notice recovery, and confirms list-only deletion
does not remove output/cache files.

Before restarting the installed service, query `/api/jobs` and avoid interrupting active jobs unless
the user explicitly requested it. After restart verify:

```bash
systemctl --user is-active lerobot-dataconvert
curl -fsS http://127.0.0.1:8765/api/health
journalctl --user -u lerobot-dataconvert -p warning --since today --no-pager
```

## Commit And Push Hygiene

- Preserve unrelated user changes and inspect all staged files before committing.
- Do not stage `.runtime`, `node_modules`, virtual environments, generated datasets, caches, or
  credentials.
- A first push should verify the destination with `git ls-remote`, add the requested remote without
  overwriting an existing one, push the current branch with upstream tracking, and verify the remote
  branch SHA afterward.
