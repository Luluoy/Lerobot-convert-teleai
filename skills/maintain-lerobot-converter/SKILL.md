---
name: maintain-lerobot-converter
description: Safely maintain the LeRobot Data Convert repository. Use for any change to Python adapters, field mappings, FPS alignment, motion filtering, multiprocessing, manifests, recovery, LeRobot output, HTTP APIs, the PWA, installation, systemd service files, tests, or repository documentation.
---

# Maintain LeRobot Converter

Last verified: 2026-07-17.

## Freshness Contract

Treat this skill as a maintained part of the product, not historical notes. Before changing the
repository, read the references relevant to the touched files. If a code, configuration, UI, API,
installation, or operational change makes any statement in this skill stale, update that statement
in the same change and set the affected document's `Last verified` date to the current date. Remove
obsolete claims instead of leaving contradictory guidance.

## Workflow

1. Read `AGENTS.md`, this file, and the relevant references below.
2. Inspect the dirty worktree and preserve unrelated user changes.
3. Trace the full path affected by the request. Dataset behavior commonly spans inspect, preview,
   motion scan, conversion workers, manifests, API serialization, and UI state.
4. Preserve the contracts documented in the references or update code, tests, and documentation
   together when intentionally changing a contract.
5. Run the smallest focused test first, then the complete verification required by
   `references/operations-and-ui.md` for the affected surface.
6. Before committing, run `git diff --check`, inspect the complete diff, and confirm no runtime
   data, credentials, generated datasets, caches, or virtual environments are staged.

## Reference Routing

- Read [architecture.md](references/architecture.md) for scheduler, worker, manifest, recovery,
  API, job deletion, state, cache, or LeRobot revision changes.
- Read [dataset-contracts.md](references/dataset-contracts.md) for adapters, raw fields, mappings,
  component names, timestamps, FPS, TeleAxis, image handling, or motion-filter changes.
- Read [operations-and-ui.md](references/operations-and-ui.md) for frontend, PWA, backend-online
  behavior, installation, systemd, tests, service restarts, or release/push work.

## Non-Negotiable Checks

- Never load TeleAxis pickle data from an untrusted source.
- Keep inspect, preview, motion pre-scan, and conversion on the same selected FPS and aligned frame
  timeline.
- Keep optional strict-zero repair field-based and episode-local. It must use every adapter-declared
  state/action field and run before both action analysis and converted-frame filtering.
- Keep trajectory task lists disjoint and exhaustive; workers must not compete for episodes.
- Preserve completed segments during stop/recovery and accept a segment only after its atomic
  completion marker exists.
- Treat task-list deletion as metadata-only by default. Do not delete source, output, or cache files
  unless the caller explicitly requests the separate cache-removal behavior.
- Repository updates must stop on any local change, persist that pause until a manual check, and use
  fast-forward-only pull. Never add automatic stash, reset, checkout, merge, or conflict resolution.
- Bump the service-worker cache key when changing cached shell assets.
- Do not commit or push until the requested verification passes.
