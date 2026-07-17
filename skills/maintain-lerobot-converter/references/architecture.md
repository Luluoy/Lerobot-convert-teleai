# Architecture And Recovery

Last verified: 2026-07-17.

Apply the freshness contract in `../SKILL.md`: update this file in the same change whenever the
implementation makes a statement below inaccurate.

## Ownership Map

- `models.py`: serialized contracts for episodes, raw fields, descriptors, frames, and job config.
- `adapters.py`: raw-data discovery, inspection, frame iteration, action-only iteration, and raw
  previews. Adapters do not own multiprocessing or LeRobot writing.
- `motion.py`: episode-local strict-zero state/action repair plus exact action-change analysis shared
  by pre-scan and conversion.
- `manager.py`: SQLite job records, config normalization, path validation, segment manifests,
  process scheduling, progress, stop/resume, recovery, and finalization orchestration.
- `conversion.py`: worker-local LeRobot v2.1 writes, completion markers, segment merging, v3.0
  packing, video preview, statistics, and atomic JSON helpers.
- `server.py`: standard-library HTTP server, API routing, static PWA delivery, and process lifecycle.
- `updates.py`: repository status inspection, persistent local-change pause, remote comparison, and
  fast-forward-only pull.
- `static/`: no-build PWA. It communicates only through the HTTP API.

## Job And Segment Lifecycle

1. Job creation inspects the source again, normalizes the config, validates paths, and creates the
   sidecar cache.
2. All segment records and their `source_indices` trajectory lists are written to `manifest.json`
   before workers start. The lists are disjoint and together cover every accepted episode.
3. Each spawned process receives only one segment's episode list and writes an isolated LeRobot
   v2.1 directory. A retry may re-run that same list after discarding its incomplete directory.
4. A segment is complete only when `.segment-complete.json` has been atomically written after its
   videos, Parquet files, and metadata.
5. Finalization merges completed v2.1 segments in source order. A v3.0 request converts that merged
   v2.1 candidate; raw episodes are not converted twice.

Do not replace preassigned lists with a shared queue from which workers race for trajectories.
Dynamic process startup limits resources and enables retry; it does not dynamically assign episode
ownership.

`field_mapping` is persisted as an ordered list of `{source, target}` rows because one raw source may
feed multiple LeRobot features. `JobConfig.from_dict()` converts the legacy source-to-target object
form into rows so existing databases and manifests remain recoverable. Never collapse rows back into
a source-keyed mapping; that silently discards repeated sources.

`fill_zero_state_action` is persisted per job and defaults to false. Workers reset its previous-value
state at every episode boundary and apply it before motion drop ranges. Do not borrow values from a
previous episode or apply it only to mapped output fields.

`cpu_limit_percent` is persisted per job, defaults to 95, and is clamped to 1-95. While a job runs,
the scheduler's duty-cycle governor pauses and resumes every active worker plus its encoder child
processes as one group. This caps aggregate worker duty across the selected core set; lowering nice
priority or reducing only the displayed percentage is not an equivalent implementation. Always
resume a process tree before termination and when the governor exits.

## State And Files

- Default job database: `~/.local/share/lerobot-dataconvert/jobs.sqlite3`.
- Git update state: `~/.local/share/lerobot-dataconvert/git-update-state.json`.
- Final output: the configured output path.
- Recovery cache: a sibling named `.<output-name>.lerobot-cache`.
- Manifest writes use temporary files, `fsync`, and `os.replace`.
- A normal stop terminates active workers, removes incomplete segment output, and leaves completed
  segments recoverable.
- Startup converts interrupted running states back to queued work after validating completion
  markers.

Never place output inside the source dataset. Do not weaken completion-marker checks or delete
completed segments during recovery.

## Job Deletion Semantics

`DELETE /api/jobs/<id>` calls `delete_job(..., remove_cache=False)`. It removes only the SQLite list
record. It must not modify the raw source, final output, recovery cache, or manifest. Active states
(`queued`, `running`, `merging`, `stopping`) cannot be deleted and must first be stopped.

`remove_cache=1` is a separate explicit operation that may remove only the sidecar cache after the
job record is eligible for deletion. It still must not remove source or final output data. UI list
deletion must not send `remove_cache=1`.

Because a metadata-only deletion preserves the sidecar, the job can later be rediscovered with the
"resume from output path" operation.

## API Boundaries

- `GET /api/health`: lightweight backend health and version.
- `GET /api/bootstrap`: adapter catalog, revisions, hardware, and job list.
- `GET /api/jobs`: current list; `GET /api/jobs/<id>` returns one record.
- `POST /api/inspect`, `/api/motion-scan`, `/api/preview/raw`, `/api/jobs`, and job stop/resume
  endpoints own their corresponding workflows.
- `POST /api/update/check` performs automatic or explicitly manual repository checks;
  `POST /api/update/pull` permits only a clean fast-forward update with no active conversion job.
- Output preview reads the final dataset after completion or a completed segment while running.
- `DELETE /api/jobs/<id>` follows the metadata-only default above.

Keep path resolution and traversal checks at every filesystem boundary. Keep API errors structured
as JSON and do not expose stored tracebacks through public job records.

## Repository Update Safety

Automatic checks inspect `git status --porcelain` before any fetch. Any tracked or untracked change
sets `paused=true` in `git-update-state.json`; later automatic checks and pull requests return that
record without running Git. Only a manual check may inspect the repository again and clear the pause
after it is clean.

Clean checks fetch the configured upstream and compare `HEAD...@{upstream}`. One-click pull is shown
only when local history has no unique commits and the upstream is ahead. The backend runs
`git pull --ff-only` with interactive credential prompts disabled and rejects pull while conversion
jobs are active. Diverged histories, detached HEAD, missing upstreams, local commits, errors, and
timeouts require technical help; never hide them with stash, reset, checkout, merge, or force.

The pull endpoint changes repository files only. It does not install dependencies or restart the
running process; the user must ask an Agent to follow `INSTALL.md` and restart the service.

## LeRobot Compatibility

The supported package is pinned to LeRobot `0.3.3`; output revisions are data format choices:

- v2.1: per-episode Parquet and video files.
- v3.0: packed Parquet/video files with episode offsets, produced from the merged v2.1 candidate.

Video encoding temporarily injects the selected CRF because LeRobot 0.3.3 does not expose encoder
options through `save_episode()`. Always restore the original encoder in `finally`.
