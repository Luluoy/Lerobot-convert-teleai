from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
import json
import os
import pickle
import re
from typing import Any, Iterator
import warnings

import cv2
import h5py
import numpy as np

from .models import DatasetDescriptor, EpisodeRef, FrameSample, RawField


_ADAPTERS: dict[str, type[RawDatasetAdapter]] = {}
_PLUGINS_LOADED = False


def register_adapter(cls: type[RawDatasetAdapter]) -> type[RawDatasetAdapter]:
    if not cls.slug:
        raise ValueError("Adapter slug must not be empty")
    _ADAPTERS[cls.slug] = cls
    return cls


def create_adapter(slug: str, source_path: str, options: dict[str, Any] | None = None) -> RawDatasetAdapter:
    _load_external_adapters()
    try:
        adapter_cls = _ADAPTERS[slug]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter: {slug}") from exc
    return adapter_cls(source_path, options or {})


def adapter_catalog() -> list[dict[str, Any]]:
    _load_external_adapters()
    return [
        {
            "slug": cls.slug,
            "name": cls.display_name,
            "description": cls.description,
            "options": cls.options_schema,
        }
        for cls in _ADAPTERS.values()
    ]


class RawDatasetAdapter(ABC):
    """Contract that any raw format implements to enter the conversion pipeline.

    Implementations only own discovery, frame iteration, and raw preview. Cache,
    multiprocessing, LeRobot writing, merging, and UI progress stay format agnostic.
    """

    slug = ""
    display_name = ""
    description = ""
    options_schema: list[dict[str, Any]] = []

    def __init__(self, source_path: str, options: dict[str, Any]):
        self.source_path = str(Path(source_path).expanduser().resolve())
        self.options = options

    @abstractmethod
    def inspect(self) -> DatasetDescriptor:
        raise NotImplementedError

    @abstractmethod
    def iter_frames(self, episode: EpisodeRef) -> Iterator[FrameSample]:
        raise NotImplementedError

    @abstractmethod
    def preview(self, episode: EpisodeRef, camera: str, frame_index: int) -> np.ndarray:
        """Return one HWC RGB uint8 frame."""
        raise NotImplementedError


def _load_external_adapters() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    modules = [item.strip() for item in os.environ.get("LEROBOT_DATACONVERT_PLUGINS", "").split(",") if item.strip()]
    for module in modules:
        import_module(module)
    try:
        plugins = entry_points(group="lerobot_dataconvert.adapters")
    except TypeError:
        plugins = entry_points().get("lerobot_dataconvert.adapters", [])
    for plugin in plugins:
        try:
            adapter = plugin.load()
            if isinstance(adapter, type) and issubclass(adapter, RawDatasetAdapter):
                register_adapter(adapter)
        except Exception as exc:
            warnings.warn(f"Could not load adapter plugin {plugin.name}: {exc}", RuntimeWarning)


@register_adapter
class HDF5JointAdapter(RawDatasetAdapter):
    slug = "hdf5_joint"
    display_name = "HDF5 Joint / RGB"
    description = "Episode HDF5 files or directories containing one HDF5 file per frame."
    options_schema = [
        {"key": "fps", "label": "FPS", "type": "number", "default": 20, "min": 1, "max": 240},
    ]

    SCHEMAS = {
        "acone_qpos": {
            "state": "observations/qpos",
            "action": "action",
            "cameras": {
                "camera_0": "observations/images/head",
                "camera_1": "observations/images/left_wrist",
                "camera_2": "observations/images/right_wrist",
            },
        },
        "joint_position": {
            "state": "puppet/joint_position",
            "action": "master/joint_position",
            "cameras": {
                "camera_0": "observations/rgb_images/camera_0",
                "camera_1": "observations/rgb_images/camera_1",
                "camera_2": "observations/rgb_images/camera_2",
            },
        },
    }

    def inspect(self) -> DatasetDescriptor:
        source = Path(self.source_path)
        episode_paths = self._list_episode_sources(source)
        if not episode_paths:
            raise ValueError(f"No HDF5 episodes found in {source}")

        episodes: list[EpisodeRef] = []
        cameras: list[str] | None = None
        camera_shapes: dict[str, tuple[int, int, int]] = {}
        state_dim = action_dim = 0
        warnings: list[str] = []

        for episode_path in episode_paths:
            probe_path = self._probe_path(episode_path)
            with h5py.File(probe_path, "r") as handle:
                schema_name, schema = self._detect_schema(handle)
                current_cameras = [
                    key for key, path in schema["cameras"].items() if path in handle
                ]
                if cameras is None:
                    cameras = current_cameras
                elif current_cameras != cameras:
                    missing = sorted(set(cameras) - set(current_cameras))
                    if missing:
                        warnings.append(f"{episode_path.name}: missing cameras {missing}")

                if not state_dim:
                    state_dim = self._feature_dim(handle[schema["state"]])
                    action_dim = self._feature_dim(handle[schema["action"]])
                frame_count = self._episode_frame_count(episode_path, handle, schema, current_cameras)

                if not camera_shapes:
                    for camera in current_cameras:
                        image = self._read_image(handle, schema["cameras"][camera], 0, episode_path.is_dir())
                        decoded = decode_image(image)
                        if decoded is not None:
                            camera_shapes[camera] = tuple(int(v) for v in decoded.shape)

            source_bytes = self._path_size(episode_path)
            episodes.append(
                EpisodeRef(
                    key=episode_path.name,
                    path=str(episode_path),
                    frame_count=frame_count,
                    source_bytes=source_bytes,
                    schema=schema_name,
                )
            )

        cameras = cameras or []
        if not cameras:
            raise ValueError("No RGB cameras found in the HDF5 schema")
        for camera in cameras:
            if camera not in camera_shapes:
                raise ValueError(f"Could not decode a preview frame for {camera}")
        if state_dim != action_dim:
            warnings.append(f"state dimension {state_dim} differs from action dimension {action_dim}")

        pixels = sum(shape[0] * shape[1] for shape in camera_shapes.values())
        memory_mb = max(768, min(4096, int(640 + pixels * 10 / (1024 * 1024))))
        fields = [
            RawField("state", (state_dim,), default_target="observation.state"),
            RawField("action", (action_dim,), default_target="action"),
        ]
        fields.extend(
            RawField(
                camera,
                camera_shapes[camera],
                dtype="uint8",
                is_image=True,
                default_target=f"observation.images.{_default_camera_name(camera)}",
            )
            for camera in cameras
        )
        return DatasetDescriptor(
            adapter=self.slug,
            source_path=self.source_path,
            episodes=episodes,
            fps=int(self.options.get("fps", 20)),
            cameras=cameras,
            camera_shapes=camera_shapes,
            state_dim=state_dim,
            action_dim=action_dim,
            source_bytes=sum(item.source_bytes for item in episodes),
            estimated_worker_memory_mb=memory_mb,
            warnings=warnings[:20],
            fields=fields,
        )

    def iter_frames(self, episode: EpisodeRef) -> Iterator[FrameSample]:
        path = Path(episode.path)
        fps = int(self.options.get("fps", 20))
        if path.is_dir():
            for frame_index, frame_path in enumerate(self._list_frame_files(path)):
                with h5py.File(frame_path, "r") as handle:
                    _, schema = self._detect_schema(handle)
                    state = np.asarray(handle[schema["state"]], dtype=np.float32).reshape(-1)
                    action = np.asarray(handle[schema["action"]], dtype=np.float32).reshape(-1)
                    images = {
                        camera: decode_image(self._read_image(handle, image_path, 0, True))
                        for camera, image_path in schema["cameras"].items()
                        if image_path in handle
                    }
                yield FrameSample(state, action, images, frame_index / fps)
            return

        with h5py.File(path, "r") as handle:
            _, schema = self._detect_schema(handle)
            state = np.asarray(handle[schema["state"]], dtype=np.float32)
            action = np.asarray(handle[schema["action"]], dtype=np.float32)
            state = state[None, :] if state.ndim == 1 else state
            action = action[None, :] if action.ndim == 1 else action
            cameras = {
                key: image_path
                for key, image_path in schema["cameras"].items()
                if image_path in handle
            }
            frame_count = min(
                [len(state), len(action)]
                + [self._dataset_frame_count(handle[path]) for path in cameras.values()]
            )
            for frame_index in range(frame_count):
                yield FrameSample(
                    np.asarray(state[frame_index], dtype=np.float32).reshape(-1),
                    np.asarray(action[frame_index], dtype=np.float32).reshape(-1),
                    {
                        camera: decode_image(self._read_image(handle, image_path, frame_index, False))
                        for camera, image_path in cameras.items()
                    },
                    frame_index / fps,
                )

    def preview(self, episode: EpisodeRef, camera: str, frame_index: int) -> np.ndarray:
        path = Path(episode.path)
        if path.is_dir():
            frames = self._list_frame_files(path)
            if not frames:
                raise ValueError(f"No frame files in {path}")
            frame_path = frames[max(0, min(frame_index, len(frames) - 1))]
            with h5py.File(frame_path, "r") as handle:
                _, schema = self._detect_schema(handle)
                image = self._read_image(handle, schema["cameras"][camera], 0, True)
        else:
            with h5py.File(path, "r") as handle:
                _, schema = self._detect_schema(handle)
                dataset = handle[schema["cameras"][camera]]
                index = max(0, min(frame_index, self._dataset_frame_count(dataset) - 1))
                image = self._read_image(handle, schema["cameras"][camera], index, False)
        decoded = decode_image(image)
        if decoded is None:
            raise ValueError(f"Could not decode {camera} frame {frame_index}")
        return decoded

    @classmethod
    def _detect_schema(cls, handle: h5py.File) -> tuple[str, dict[str, Any]]:
        for name, schema in cls.SCHEMAS.items():
            if schema["state"] in handle and schema["action"] in handle:
                return name, schema
        raise ValueError("Unsupported HDF5 schema")

    @classmethod
    def _list_episode_sources(cls, source: Path) -> list[Path]:
        if source.is_file() and source.suffix.lower() in {".h5", ".hdf5"}:
            return [source]
        if not source.is_dir():
            return []
        files = [path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}]
        if files:
            return sorted(files, key=natural_key)
        directories = [path for path in source.iterdir() if path.is_dir() and cls._list_frame_files(path)]
        return sorted(directories, key=natural_key)

    @staticmethod
    def _list_frame_files(path: Path) -> list[Path]:
        return sorted(
            [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".h5", ".hdf5"}],
            key=natural_key,
        )

    @classmethod
    def _probe_path(cls, episode_path: Path) -> Path:
        return cls._list_frame_files(episode_path)[0] if episode_path.is_dir() else episode_path

    @classmethod
    def _episode_frame_count(
        cls, episode_path: Path, handle: h5py.File, schema: dict[str, Any], cameras: list[str]
    ) -> int:
        if episode_path.is_dir():
            return len(cls._list_frame_files(episode_path))
        state = handle[schema["state"]]
        action = handle[schema["action"]]
        lengths = [1 if state.ndim == 1 else len(state), 1 if action.ndim == 1 else len(action)]
        lengths.extend(cls._dataset_frame_count(handle[schema["cameras"][key]]) for key in cameras)
        return min(lengths)

    @staticmethod
    def _feature_dim(dataset: h5py.Dataset) -> int:
        return int(dataset.shape[-1]) if dataset.ndim > 1 else int(dataset.size)

    @staticmethod
    def _dataset_frame_count(dataset: h5py.Dataset) -> int:
        if dataset.ndim >= 4:
            return len(dataset)
        if dataset.ndim == 2:
            return len(dataset)
        if dataset.ndim == 1 and dataset.dtype.kind in {"O", "V"}:
            return len(dataset)
        return 1

    @staticmethod
    def _read_image(handle: h5py.File, dataset_path: str, frame_index: int, single_file: bool) -> Any:
        if dataset_path not in handle:
            return None
        dataset = handle[dataset_path]
        if single_file:
            if dataset.ndim >= 4 or dataset.ndim == 2 or (dataset.ndim == 1 and dataset.dtype.kind in {"O", "V"}):
                return dataset[0]
            return dataset[()]
        if HDF5JointAdapter._dataset_frame_count(dataset) == 1:
            return dataset[()]
        return dataset[frame_index]

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


@register_adapter
class MultiProcessingPoolDatasetAdapter(RawDatasetAdapter):
    slug = "multiprocessing_pool_dataset"
    display_name = "MultiProcessing Pool Dataset"
    description = "Trusted TeleAxis Collector schema-v3 PKL streams and PNG cameras."

    _EPISODE_RE = re.compile(r"episode_(\d+)$")
    _FRAME_RE = re.compile(r"frame_(\d{9})_(\d+)\.pkl$")

    def inspect(self) -> DatasetDescriptor:
        source = Path(self.source_path)
        episode_paths = self._episode_paths(source)
        if not episode_paths:
            raise ValueError(f"No TeleAxis Collector episodes found in {source}")

        episodes: list[EpisodeRef] = []
        warnings: list[str] = []
        reference: dict[str, Any] | None = None
        for episode_path in episode_paths:
            try:
                info = self._inspect_episode(episode_path)
                if reference is None:
                    reference = info
                else:
                    self._require_matching_layout(reference, info, episode_path)
            except Exception as exc:
                warnings.append(f"{episode_path.name}: skipped ({exc})")
                continue

            episodes.append(
                EpisodeRef(
                    key=episode_path.name,
                    path=str(episode_path),
                    frame_count=info["frame_count"],
                    source_bytes=info["source_bytes"],
                    schema="teleaxis_collector_v3",
                )
            )
            validation = info["meta"].get("validation")
            status = str(validation.get("status", "")) if isinstance(validation, Mapping) else ""
            if status.upper() not in {"", "PASS", "SKIPPED", "NOT_RUN"}:
                warnings.append(f"{episode_path.name}: validation status is {status}")

        if not episodes or reference is None:
            detail = "; ".join(warnings[:3])
            raise ValueError(f"No complete TeleAxis Collector schema-v3 episodes found. {detail}")

        pixels = sum(shape[0] * shape[1] for shape in reference["camera_shapes"].values())
        memory_mb = max(768, min(4096, int(640 + pixels * 10 / (1024 * 1024))))
        return DatasetDescriptor(
            adapter=self.slug,
            source_path=self.source_path,
            episodes=episodes,
            fps=reference["fps"],
            cameras=reference["cameras"],
            camera_shapes=reference["camera_shapes"],
            state_dim=reference["state_dim"],
            action_dim=reference["action_dim"],
            source_bytes=sum(episode.source_bytes for episode in episodes),
            estimated_worker_memory_mb=memory_mb,
            warnings=warnings[:20],
            fields=reference["fields"],
        )

    def iter_frames(self, episode: EpisodeRef) -> Iterator[FrameSample]:
        episode_path = Path(episode.path)
        meta = self._meta(episode_path)
        state_meta = self._stream_meta(meta, "joint_state", "state")
        action_meta = self._stream_meta(meta, "joint_action", "action")
        eef_meta = self._stream_meta(meta, "eef_action", "action")
        cameras = self._camera_names(meta)
        state_files = self._frame_files(episode_path, state_meta, episode.frame_count)
        action_files = self._frame_files(episode_path, action_meta, episode.frame_count)
        eef_files = self._frame_files(episode_path, eef_meta, episode.frame_count)
        if [path.name for path in action_files] != [path.name for path in state_files]:
            raise ValueError("joint_action frames do not align with joint_state")
        if [path.name for path in eef_files] != [path.name for path in state_files]:
            raise ValueError("eef_action frames do not align with joint_state")
        episode_index = int(meta["episode_index"])
        first_monotonic_ns: int | None = None

        for state_path, action_path, eef_path in zip(
            state_files, action_files, eef_files, strict=True
        ):
            state_record = self._record(state_path, episode_index)
            action_record = self._record(action_path, episode_index)
            eef_record = self._record(eef_path, episode_index)
            qpos = self._vector(state_record, "qpos")
            joint_action = self._vector(action_record, "action")
            if qpos.shape != joint_action.shape or not np.array_equal(qpos, joint_action):
                raise ValueError(f"joint action does not equal qpos in {state_path.name}")
            images = {
                camera: self._camera_image(episode_path, meta, state_record, camera)
                for camera in cameras
            }
            monotonic_ns = self._integer(state_record, "monotonic_timestamp_ns")
            if first_monotonic_ns is None:
                first_monotonic_ns = monotonic_ns
            timestamp = max(0.0, (monotonic_ns - first_monotonic_ns) / 1e9)
            yield FrameSample(
                state=qpos,
                action=joint_action,
                images=images,
                timestamp=timestamp,
                fields={
                    "joint_state/qpos": qpos,
                    "joint_state/qvel": self._vector(state_record, "qvel"),
                    "joint_state/torque": self._vector(state_record, "torque"),
                    "joint_action/action": joint_action,
                    "eef_action/action": self._vector(eef_record, "action"),
                    **images,
                },
            )

    def preview(self, episode: EpisodeRef, camera: str, frame_index: int) -> np.ndarray:
        episode_path = Path(episode.path)
        meta = self._meta(episode_path)
        if camera not in self._camera_names(meta):
            raise KeyError(f"Unknown camera field: {camera}")
        state_meta = self._stream_meta(meta, "joint_state", "state")
        state_files = self._frame_files(episode_path, state_meta, episode.frame_count)
        index = max(0, min(int(frame_index), len(state_files) - 1))
        record = self._record(state_files[index], int(meta["episode_index"]))
        return self._camera_image(episode_path, meta, record, camera)

    def _inspect_episode(self, episode_path: Path) -> dict[str, Any]:
        meta = self._meta(episode_path)
        if str(meta.get("status", "")).lower() != "complete":
            raise ValueError(f"episode status is {meta.get('status', 'missing')}")
        save_errors = meta.get("save_errors", [])
        if not isinstance(save_errors, list) or save_errors:
            raise ValueError("episode has save errors")
        frame_count = self._integer(meta, "frame_count")
        saved_frame_count = self._integer(meta, "saved_frame_count")
        if frame_count <= 0 or saved_frame_count != frame_count:
            raise ValueError(f"frame count mismatch ({frame_count}/{saved_frame_count})")

        state_meta = self._stream_meta(meta, "joint_state", "state")
        action_meta = self._stream_meta(meta, "joint_action", "action")
        eef_meta = self._stream_meta(meta, "eef_action", "action")
        state_names = self._field_names(state_meta, "joint_state")
        action_names = self._field_names(action_meta, "joint_action")
        eef_names = self._field_names(eef_meta, "eef_action")
        state_files = self._frame_files(episode_path, state_meta, frame_count)
        action_files = self._frame_files(episode_path, action_meta, frame_count)
        eef_files = self._frame_files(episode_path, eef_meta, frame_count)
        if [path.name for path in action_files] != [path.name for path in state_files]:
            raise ValueError("joint_action frames do not align with joint_state")
        if [path.name for path in eef_files] != [path.name for path in state_files]:
            raise ValueError("eef_action frames do not align with joint_state")
        first_state = self._record(state_files[0], int(meta["episode_index"]))
        first_action = self._record(action_files[0], int(meta["episode_index"]))
        first_eef = self._record(eef_files[0], int(meta["episode_index"]))
        for field_name in ("qpos", "qvel", "torque"):
            if self._vector(first_state, field_name).size != len(state_names):
                raise ValueError(f"joint_state/{field_name} does not match META order")
        if self._vector(first_action, "action").size != len(action_names):
            raise ValueError("joint_action/action does not match META order")
        if self._vector(first_eef, "action").size != len(eef_names):
            raise ValueError("eef_action/action does not match META order")

        cameras = self._camera_names(meta)
        camera_shapes = {
            camera: tuple(
                int(size)
                for size in self._camera_image(episode_path, meta, first_state, camera).shape
            )
            for camera in cameras
        }
        fields = [
            RawField(
                "joint_state/qpos",
                (len(state_names),),
                default_target="observation.state",
                names=state_names,
            ),
            RawField("joint_state/qvel", (len(state_names),), names=state_names),
            RawField("joint_state/torque", (len(state_names),), names=state_names),
            RawField(
                "joint_action/action",
                (len(action_names),),
                default_target="action",
                names=action_names,
            ),
            RawField("eef_action/action", (len(eef_names),), names=eef_names),
        ]
        streams = meta["streams"]
        for camera in cameras:
            stream = streams[camera]
            camera_name = str(stream.get("camera_name") or camera)
            fields.append(
                RawField(
                    camera,
                    camera_shapes[camera],
                    dtype="uint8",
                    is_image=True,
                    default_target=f"observation.images.{_safe_field_part(camera_name)}",
                )
            )

        fps_value = state_meta.get("nominal_fps")
        if isinstance(fps_value, bool):
            raise ValueError("joint_state nominal_fps is invalid")
        fps = int(round(float(fps_value)))
        if fps <= 0:
            raise ValueError("joint_state nominal_fps must be positive")
        return {
            "meta": meta,
            "frame_count": frame_count,
            "fps": fps,
            "cameras": cameras,
            "camera_shapes": camera_shapes,
            "state_dim": len(state_names),
            "action_dim": len(action_names),
            "fields": fields,
            "source_bytes": self._tree_size(episode_path),
        }

    @staticmethod
    def _require_matching_layout(reference: dict[str, Any], current: dict[str, Any], path: Path) -> None:
        keys = ("fps", "cameras", "camera_shapes", "state_dim", "action_dim", "fields")
        differences = [key for key in keys if current[key] != reference[key]]
        if differences:
            raise ValueError(f"layout differs in {', '.join(differences)}")

    @classmethod
    def _episode_paths(cls, source: Path) -> list[Path]:
        if (source / "META" / "meta.json").is_file():
            return [source]
        if not source.is_dir():
            return []
        return sorted(
            [path for path in source.iterdir() if path.is_dir() and cls._EPISODE_RE.fullmatch(path.name)],
            key=natural_key,
        )

    @staticmethod
    def _meta(episode_path: Path) -> dict[str, Any]:
        path = episode_path / "META" / "meta.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("META/meta.json must contain an object")
        if value.get("schema") != "teleaxis_collector_episode" or value.get("schema_version") != 3:
            raise ValueError("unsupported TeleAxis Collector schema")
        return value

    @staticmethod
    def _stream_meta(meta: Mapping[str, Any], name: str, expected_type: str) -> Mapping[str, Any]:
        streams = meta.get("streams")
        stream = streams.get(name) if isinstance(streams, Mapping) else None
        if not isinstance(stream, Mapping) or stream.get("type") != expected_type:
            raise ValueError(f"missing {expected_type} stream {name}")
        return stream

    @staticmethod
    def _field_names(stream: Mapping[str, Any], name: str) -> tuple[str, ...]:
        order = stream.get("order")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)) or not order:
            raise ValueError(f"{name}.order must be a non-empty sequence")
        return tuple(str(value) for value in order)

    @classmethod
    def _frame_files(
        cls, episode_path: Path, stream: Mapping[str, Any], expected_count: int
    ) -> list[Path]:
        directory = cls._stream_path(episode_path, stream)
        files = [path for path in directory.iterdir() if path.is_file() and path.suffix == ".pkl"]
        identities: list[tuple[int, int, Path]] = []
        for path in files:
            match = cls._FRAME_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"invalid frame filename {path.name}")
            identities.append((int(match.group(1)), int(match.group(2)), path))
        identities.sort(key=lambda item: (item[0], item[1]))
        indices = [item[0] for item in identities]
        if indices != list(range(expected_count)):
            label = str(stream.get("path") or "stream")
            raise ValueError(f"{label} frame indices are not contiguous ({len(indices)}/{expected_count})")
        return [item[2] for item in identities]

    @staticmethod
    def _stream_path(episode_path: Path, stream: Mapping[str, Any]) -> Path:
        raw_path = stream.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("stream path is missing")
        return MultiProcessingPoolDatasetAdapter._within(episode_path, episode_path / raw_path)

    @classmethod
    def _record(cls, path: Path, episode_index: int) -> dict[str, Any]:
        match = cls._FRAME_RE.fullmatch(path.name)
        if match is None or not path.is_file():
            raise ValueError(f"missing frame record {path}")
        value = pickle.loads(path.read_bytes())
        if not isinstance(value, Mapping):
            raise ValueError(f"{path.name} must contain a mapping")
        record = dict(value)
        if record.get("schema_version") != 3:
            raise ValueError(f"{path.name} has an unsupported schema")
        if cls._integer(record, "episode_index") != episode_index:
            raise ValueError(f"{path.name} episode_index mismatch")
        if cls._integer(record, "frame_index") != int(match.group(1)):
            raise ValueError(f"{path.name} frame_index mismatch")
        if cls._integer(record, "timestamp_ns") != int(match.group(2)):
            raise ValueError(f"{path.name} timestamp mismatch")
        cls._integer(record, "monotonic_timestamp_ns")
        return record

    @staticmethod
    def _vector(record: Mapping[str, Any], name: str) -> np.ndarray:
        value = np.asarray(record.get(name), dtype=np.float32)
        if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a non-empty finite vector")
        return value

    @staticmethod
    def _integer(record: Mapping[str, Any], name: str) -> int:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    @staticmethod
    def _camera_names(meta: Mapping[str, Any]) -> list[str]:
        streams = meta.get("streams")
        if not isinstance(streams, Mapping):
            raise ValueError("META streams must be a mapping")
        cameras = [
            str(name)
            for name, value in streams.items()
            if isinstance(value, Mapping) and value.get("type") == "camera"
        ]
        if not cameras:
            raise ValueError("episode declares no camera streams")
        return cameras

    @classmethod
    def _camera_image(
        cls,
        episode_path: Path,
        meta: Mapping[str, Any],
        state_record: Mapping[str, Any],
        camera: str,
    ) -> np.ndarray:
        cameras = state_record.get("cameras")
        value = cameras.get(camera) if isinstance(cameras, Mapping) else None
        if not isinstance(value, Mapping):
            raise ValueError(f"joint_state is missing camera {camera}")
        image_value = value.get("image_path")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError(f"camera {camera} image_path is missing")
        stream = cls._stream_meta(meta, camera, "camera")
        camera_dir = cls._stream_path(episode_path, stream)
        image_path = cls._within(camera_dir, episode_path / image_value)
        decoded = decode_image(image_path.read_bytes())
        if decoded is None:
            raise ValueError(f"could not decode camera image {image_path}")
        height = int(stream.get("height", decoded.shape[0]))
        width = int(stream.get("width", decoded.shape[1]))
        if decoded.shape != (height, width, 3):
            raise ValueError(f"camera {camera} shape {decoded.shape} != {(height, width, 3)}")
        return decoded

    @staticmethod
    def _within(root: Path, path: Path) -> Path:
        root = root.resolve()
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes dataset root: {path}") from exc
        return path

    @staticmethod
    def _tree_size(path: Path) -> int:
        total = 0
        for directory, _, names in os.walk(path):
            for name in names:
                total += (Path(directory) / name).stat().st_size
        return total


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def _default_camera_name(camera: str) -> str:
    match = re.search(r"(\d+)$", camera)
    return f"image_{match.group(1)}" if match else _safe_field_part(camera)


def _safe_field_part(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_").lower()
    return result or "image"


def decode_image(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(value, dtype=np.uint8)
    else:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] >= 3:
            return np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)
        if array.dtype == object and array.size == 1:
            item = array.item()
            encoded = np.frombuffer(item, dtype=np.uint8) if isinstance(item, (bytes, bytearray, memoryview)) else np.asarray(item, dtype=np.uint8).reshape(-1)
        else:
            encoded = array.astype(np.uint8, copy=False).reshape(-1)
    if encoded.size == 0:
        return None
    if encoded.size > 4 and encoded[0] == 0xFF and encoded[1] == 0xD8:
        endings = np.flatnonzero((encoded[:-1] == 0xFF) & (encoded[1:] == 0xD9))
        if endings.size:
            encoded = encoded[: endings[-1] + 2]
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def jpeg_bytes(image: np.ndarray, quality: int = 88) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode preview JPEG")
    return encoded.tobytes()
