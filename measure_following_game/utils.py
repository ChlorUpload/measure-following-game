# -*- coding: utf-8 -*-

__all__ = ["make_env_base_param", "make_env_component_param"]

from beartype import beartype
from sabanamusic.common.types import PathLike, PositiveInt

from measure_following_game.params import *
from measure_following_game.params import EnvComponentParam


@beartype
def make_env_base_param(
    score_root: PathLike,
    record_name: str | None = None,
    window_size: PositiveInt = 32,
    memory_size: PositiveInt = 32,
    fps: PositiveInt = 20,
    onset_only: bool = True,
) -> EnvBaseParam:
    return EnvBaseParam(
        score_root=str(score_root),
        record_name=record_name,
        window_size=window_size,
        memory_size=memory_size,
        fps=fps,
        onset_only=onset_only,
    )


@beartype
def make_env_component_param(
    reward_id: str = "Triangle",
    manager_id: str = "MIDI",
    renderer_id: str = "Grid",
    record_id: str = "MIDI",
) -> EnvComponentParam:
    return EnvComponentParam(
        reward_id=reward_id,
        manager_id=manager_id,
        record_id=record_id,
        renderer_id=renderer_id,
    )
