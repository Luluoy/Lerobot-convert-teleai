# Dataset Contracts

Last verified: 2026-07-16.

Apply the freshness contract in `../SKILL.md`: update this file in the same change whenever the
implementation makes a statement below inaccurate.

## Adapter Contract

Every `RawDatasetAdapter` implements:

- `inspect() -> DatasetDescriptor`
- `iter_frames(EpisodeRef) -> Iterator[FrameSample]`
- `preview(EpisodeRef, camera, frame_index) -> RGB uint8 HWC image`

`iter_action_values()` defaults to `iter_frames()` but should be overridden when action streams can
be read without decoding images. Register built-ins with `@register_adapter`; external adapters use
`LEROBOT_DATACONVERT_PLUGINS` or the `lerobot_dataconvert.adapters` entry-point group.

`DatasetDescriptor.fields` is the source of truth. Each `RawField` declares name, shape, dtype,
image status, default target, component names, action status, and native FPS. `FrameSample.fields`
must contain the same source names. Legacy `state`, `action`, and camera members remain compatibility
fallbacks.

## Field Mapping And Names

- Mapping changes destination feature names, not source values or ordering.
- Target names must be unique, valid, and not reserved LeRobot fields.
- Images cannot map to numeric state/action fields; numeric values cannot map under
  `observation.images.*`.
- Leaving a target blank ignores that source field.
- `observation.state names override` and `action names override` affect component metadata only.
  They do not reorder, select, or transform data. When the supplied count differs from the feature
  dimension, the converter falls back to generated `state_N` or `action_N` names.
- Motion analysis uses every field marked `is_action=True`, including fields not mapped into final
  output.

## TeleAxis MultiProcessing Pool Dataset

Only accept TeleAxis Collector `schema_version: 3` episodes with status `complete`, equal positive
`frame_count` and `saved_frame_count`, and no save errors. The required numeric streams are:

- `joint_state`: `qpos`, `qvel`, `torque`
- `joint_action`: `action`
- `eef_action`: `action`

Camera streams are PNG directories declared by META. Stream paths and individual files must remain
inside the episode/stream root. Pickle is code-execution capable; load only trusted locally produced
collector datasets.

Each field's FPS uses a positive `actual_fps` when present, otherwise positive `nominal_fps`. Across
accepted episodes, the descriptor stores the minimum rate observed for each field. The selected
target FPS is a positive integer no higher than the global minimum field FPS; blank/zero means
automatic floor of that minimum.

Use wall-clock `timestamp_ns` for cross-sensor alignment because PNG filenames contain camera wall
clock timestamps. Do not align cameras against PKL-only `monotonic_timestamp_ns`. Build the target
timeline over the intersection of every required stream's available wall-clock range. At each target
trigger choose each stream's nearest sample; ties choose the earlier timestamp. Preview, action scan,
and frame conversion must use this identical timeline and frame count.

Independent streams may have different counts and non-contiguous frame indices. Do not restore the
old assumption that PKL and camera filenames must share one frame index or filename.

## Motion Filtering

Action equality currently uses exact `numpy.array_equal` across all declared action fields. The
"continuous stationary threshold" is a frame-count threshold, not a numeric movement tolerance.

- Start trimming removes the unchanged prefix while retaining at least one frame.
- Stationary-segment removal keeps the first frame of a stationary run and removes later redundant
  frames when the run meets the configured threshold.
- Every episode must still emit at least one frame.
- Pre-scan and conversion must call the same analysis functions so counts and removed duration agree.

If numeric tolerance is introduced later, update the UI wording, API/config, analysis, tests, and
this document together.

## Image And Video Rules

Adapters return RGB `uint8` HWC arrays. Conversion trims extra channels, resizes to the declared
shape when necessary, and writes mapped images as LeRobot video features. Raw preview and output
preview must both expose real data. Preserve the selected AV1 CRF through every worker and revision.
