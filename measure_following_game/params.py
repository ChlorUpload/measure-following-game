# -*- coding: utf-8 -*-

__all__ = ["EnvParam"]

from dataclasses import dataclass
from dataclasses_io import dataclass_io


@dataclass_io
@dataclass
class EnvParam:
    score_root: str
    record_name: str | None = None
    window_size: int = 32
    memory_size: int = 32
    fps: int = 20
    onset_only: bool = True


if __name__ == "__main__":
    param = EnvParam(score_root="")
    print(param.config)
    param.save_json("./param.json")
