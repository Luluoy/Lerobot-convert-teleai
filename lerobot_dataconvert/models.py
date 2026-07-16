from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodeRef:
    key: str
    path: str
    frame_count: int
    source_bytes: int = 0
    schema: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeRef":
        return cls(**value)


@dataclass(frozen=True)
class RawField:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    is_image: bool = False
    default_target: str = ""
    names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RawField":
        value = dict(value)
        value["shape"] = tuple(int(size) for size in value["shape"])
        value["names"] = tuple(str(name) for name in value.get("names", ()))
        return cls(**value)


@dataclass
class DatasetDescriptor:
    adapter: str
    source_path: str
    episodes: list[EpisodeRef]
    fps: int
    cameras: list[str]
    camera_shapes: dict[str, tuple[int, int, int]]
    state_dim: int
    action_dim: int
    source_bytes: int
    estimated_worker_memory_mb: int
    warnings: list[str] = field(default_factory=list)
    fields: list[RawField] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(episode.frame_count for episode in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_frames"] = self.total_frames
        return data

    def resolved_fields(self) -> list[RawField]:
        if self.fields:
            return list(self.fields)
        fields = [
            RawField("state", (self.state_dim,), default_target="observation.state"),
            RawField("action", (self.action_dim,), default_target="action"),
        ]
        fields.extend(
            RawField(
                camera,
                self.camera_shapes[camera],
                dtype="uint8",
                is_image=True,
                default_target=f"observation.images.{camera}",
            )
            for camera in self.cameras
        )
        return fields

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetDescriptor":
        value = dict(value)
        value.pop("total_frames", None)
        value["episodes"] = [EpisodeRef.from_dict(item) for item in value["episodes"]]
        value["camera_shapes"] = {
            key: tuple(shape) for key, shape in value["camera_shapes"].items()
        }
        value["fields"] = [RawField.from_dict(item) for item in value.get("fields", [])]
        return cls(**value)


@dataclass
class FrameSample:
    state: Any
    action: Any
    images: dict[str, Any]
    timestamp: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def as_fields(self) -> dict[str, Any]:
        values = dict(self.fields)
        if self.state is not None:
            values.setdefault("state", self.state)
        if self.action is not None:
            values.setdefault("action", self.action)
        for name, image in self.images.items():
            values.setdefault(name, image)
        return values


@dataclass
class JobConfig:
    adapter: str
    source_path: str
    output_path: str
    revision: str
    repo_id: str
    robot_type: str
    task_instruction: str
    fps: int
    cpu_cores: int
    memory_gb: float
    segment_size: int
    camera_names: dict[str, str]
    state_names: list[str]
    action_names: list[str]
    field_mapping: dict[str, str] = field(default_factory=dict)
    adapter_options: dict[str, Any] = field(default_factory=dict)
    skip_zero_state: bool = True
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobConfig":
        return cls(**value)
