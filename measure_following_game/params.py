# -*- coding: utf-8 -*-

__all__ = ["EnvBaseParam", "EnvComponentParam"]

from dataclasses import dataclass
from dataclasses_io import dataclass_io


@dataclass_io
@dataclass
class EnvBaseParam:
    score_root: str
    record_name: str | None = None
    window_size: int = 32
    memory_size: int = 32
    fps: int = 20
    onset_only: bool = True


@dataclass_io
@dataclass
class EnvComponentParam:
    reward_id: str = "Triangle"
    manager_id: str = "MIDI"
    renderer_id: str = "Grid"
    record_id: str = "MIDI"
