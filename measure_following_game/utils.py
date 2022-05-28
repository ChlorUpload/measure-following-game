# -*- coding: utf-8 -*-

__all__ = ["make_env_param"]

from beartype import beartype
from sabanamusic.common.types import PathLike, PositiveInt

from measure_following_game.params import *


@beartype
def make_env_param(
    score_root: PathLike,
    record_name: str | None = None,
    window_size: PositiveInt = 32,
    memory_size: PositiveInt = 32,
    fps: PositiveInt = 20,
    onset_only: bool = True,
) -> EnvParam:
    return EnvParam(
        score_root=str(score_root),
        record_name=record_name,
        window_size=window_size,
        memory_size=memory_size,
        fps=fps,
        onset_only=onset_only,
    )
