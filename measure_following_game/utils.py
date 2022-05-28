# -*- coding: utf-8 -*-

__all__ = ["make_env_base_param", "make_env_component_param"]

from pathlib import Path
import random

from beartype import beartype
from sabanamusic.common.types import PathLike, PositiveInt
from sabanamusic.models.musical import Record, MIDIRecord

from measure_following_game.params import *
from measure_following_game.environment import *


@beartype
def make_env_base_param(
    score_root: PathLike,
    record_name: str | None = None,
    window_size: PositiveInt = 32,
    memory_size: PositiveInt = 32,
    fps: PositiveInt = 20,
    onset_only: bool = True,
) -> EnvBaseParam:
    score_root = str(Path(score_root).resolve())
    return EnvBaseParam(
        score_root=score_root,
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


# TODO(kaparoo): need factory class method for Reward, ContextManager, Record, and Renderer?


@beartype
def make_reward(reward_id: str, window_size: int):
    match reward_id.lower():
        case "triangle":
            return TriangleReward(window_size)
        case _:
            raise KeyError(f"Unknown id: {reward_id}")


@beartype
def make_record(
    record_id: str,
    record_root: PathLike,
    fps: PositiveInt = 20,
    duration: PositiveInt = 3,
    step_freq: PositiveInt = 10,
    onset_only: bool = True,
):
    match record_id.lower():
        case "midi":
            return MIDIRecord(record_root, fps, duration, step_freq, onset_only)
        case _:
            raise KeyError(f"Unknown id: {record_id}")


@beartype
def make_renderer(
    renderer_id: str,
    score_root: PathLike,
    fps: PositiveInt = 20,
    onset_only: bool = True,
):
    match renderer_id.lower():
        case "grid":
            return GridContextRenderer(score_root, fps, onset_only)
        case _:
            raise KeyError(f"Unknown id: {renderer_id}")


@beartype
def make_manager(
    manager_id: str,
    renderer: ContextRenderer,
    record: Record,
    window_size: PositiveInt = 32,
    memory_size: PositiveInt = 32,
):
    match manager_id.lower():
        case "midi":
            return MIDIContextManager(renderer, record, window_size, memory_size)
        case _:
            raise KeyError(f"Unknown id: {manager_id}")


@beartype
def get_valid_midi_record_root(
    score_root: PathLike, record_name: str | None = None
) -> Path:
    score_root = Path(score_root)

    if record_name is None:
        record_name = ""

    record_root = score_root / record_name

    def is_valid_record(record_root: Path) -> bool:
        record_midi_path = record_root.with_suffix(".midi")
        record_annotation_path = record_root.with_suffix(".csv")
        return record_midi_path.exists() and record_annotation_path.exists()

    if not is_valid_record(record_root):
        valid_record_roots = [
            record.with_suffix("")
            for record in score_root.rglob("**/*.midi")
            if "score" not in record and is_valid_record(record.with_suffix(""))
        ]
        if not valid_record_roots:
            raise FileNotFoundError(f"cannot find any MIDI record in {str(score_root)}")
        record_root = random.choice(valid_record_roots)

    record_name = str(record_root.resolve()).replace(str(score_root.resolve()))

    return record_root


@beartype
def make_env(
    base_param: EnvBaseParam,
    component_param: EnvComponentParam,
    duration: PositiveInt = 3,
    step_freq: PositiveInt = 10,
) -> MeasureFollowingEnv:
    reward = make_reward(component_param.reward_id, base_param.window_size)

    renderer = make_renderer(
        component_param.renderer_id,
        base_param.score_root,
        base_param.fps,
        base_param.onset_only,
    )

    record_root, record_name = get_valid_midi_record_root(
        base_param.score_root, base_param.record_name
    )
    base_param.record_name = record_name

    record = make_record(
        component_param.record_id,
        record_root,
        base_param.fps,
        duration,
        step_freq,
        base_param.onset_only,
    )

    manager = make_manager(
        component_param.manager_id,
        renderer,
        record,
        base_param.window_size,
        base_param.memory_size,
    )

    return MeasureFollowingEnv(manager, reward)
