from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
import os
import re
from typing import Any, Iterator
import warnings

import cv2
import h5py
import numpy as np

from .models import DatasetDescriptor, EpisodeRef, FrameSample


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


def natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


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
