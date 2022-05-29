# -*- coding: utf-8 -*-

__all__ = ["make_env_param"]

from typing import Any

from beartype import beartype
from sabanamusic.common.types import PathLike, PositiveInt
from sabanamusic.models.musical import Record, MIDIRecord

from measure_following_game.params import *
from measure_following_game.environment import *


@beartype
def make_env_param(
    # commons
    score_root: PathLike,
    record_name: str | None = None,
    window_size: PositiveInt = 32,
    memory_size: PositiveInt = 32,
    fps: PositiveInt = 20,
    onset_only: bool = True,
    buffer_duration: PositiveInt = 3,
    buffer_step_size: PositiveInt = 10,
    # component sepecifics
    reward_id: str = "triangle",
    manager_id: str = "midi",
    renderer_id: str = "grid",
    record_id: str = "midi",
    reward_options: dict[str, Any] = {},
    record_options: dict[str, Any] = {},
    renderer_options: dict[str, Any] = {},
    manager_options: dict[str, Any] = {},
) -> EnvParam:
    return EnvParam(
        score_root,
        record_name,
        window_size,
        memory_size,
        fps,
        onset_only,
        buffer_duration,
        buffer_step_size,
        reward_id,
        manager_id,
        renderer_id,
        record_id,
        reward_options,
        record_options,
        renderer_options,
        manager_options,
    )


# TODO(kaparoo): need factory class method for Reward, Record, Renderer, and Manager?


@beartype
def make_reward(reward_id: str, window_size: int, reward_options: dict = {}):
    match reward_id.lower():
        case "triangle":
            return TriangleReward(window_size, **reward_options)
        case _:
            raise KeyError(f"Unknown id: {reward_id}")


@beartype
def make_record(
    record_id: str,
    score_root: PathLike,
    record_name: str | None = None,
    fps: PositiveInt = 20,
    duration: PositiveInt = 3,
    step_freq: PositiveInt = 10,
    onset_only: bool = True,
    record_options: dict = {},
):
    match record_id.lower():
        case "midi":
            record_root, _ = MIDIRecord.get_valid_record_root(score_root, record_name)
            return MIDIRecord(
                record_root, fps, duration, step_freq, onset_only, **record_options
            )
        case _:
            raise KeyError(f"Unknown id: {record_id}")


@beartype
def make_renderer(
    renderer_id: str,
    score_root: PathLike,
    fps: PositiveInt = 20,
    onset_only: bool = True,
    renderer_options: dict = {},
):
    match renderer_id.lower():
        case "grid":
            return GridContextRenderer(score_root, fps, onset_only, **renderer_options)
        case _:
            raise KeyError(f"Unknown id: {renderer_id}")


@beartype
def make_manager(
    manager_id: str,
    renderer: ContextRenderer,
    record: Record,
    window_size: PositiveInt = 32,
    memory_size: PositiveInt = 32,
    manager_options: dict = {},
):
    match manager_id.lower():
        case "midi":
            return MIDIContextManager(
                renderer, record, window_size, memory_size, **manager_options
            )
        case _:
            raise KeyError(f"Unknown id: {manager_id}")


@beartype
def make_env(param: EnvParam) -> MeasureFollowingEnv:
    reward = make_reward(param.reward_id, param.window_size)

    renderer = make_renderer(
        param.renderer_id, param.score_root, param.fps, param.onset_only
    )

    record = make_record(
        param.record_id,
        param.score_root,
        param.record_name,
        param.fps,
        param.buffer_duration,
        param.buffer_step_size,
        param.onset_only,
    )

    manager = make_manager(
        param.manager_id,
        renderer,
        record,
        param.window_size,
        param.memory_size,
    )

    return MeasureFollowingEnv(manager, reward)
