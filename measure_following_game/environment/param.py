# -*- coding: utf-8 -*-

__all__ = ["EnvParam"]

from dataclasses import dataclass, field
from typing import Any

from dataclasses_io import dataclass_io


@dataclass_io
@dataclass
class EnvParam:
    # common
    score_root: str
    record_name: str | None = None
    window_size: int = 16
    memory_size: int = 16
    fps: int = 20
    onset_only: bool = True
    buffer_duration: int = 3
    buffer_step_size: int = 10

    # component
    reward_id: str = "triangle"
    record_id: str = "midi"
    renderer_id: str = "grid"
    manager_id: str = "midi"

    reward_options: dict[str, Any] = field(default_factory=dict)
    record_options: dict[str, Any] = field(default_factory=dict)
    renderer_options: dict[str, Any] = field(default_factory=dict)
    manager_options: dict[str, Any] = field(default_factory=dict)
