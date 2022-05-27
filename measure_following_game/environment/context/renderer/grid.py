# -*- coding: utf-8 -*-

__all__ = ["GridContextRenderer"]

from typing import ClassVar, Literal

from beartype import beartype
from sabanamusic.common.types import Index, PathLike, PositiveInt

from measure_following_game.environment.context.renderer.base import ContextRenderer
from measure_following_game.types import ActType


class GridContextRenderer(ContextRenderer):

    modes: ClassVar[list[str]] = ["human", "rgb_array"]

    @beartype
    def __init__(
        self,
        score_root: PathLike,
        fps: PositiveInt = 20,
        onset_only: bool = True,
    ):
        super().__init__(score_root, fps, onset_only)

    def slide(self):
        raise NotImplementedError()

    @beartype
    def step(self, pred_policy: ActType):
        raise NotImplementedError()

    @beartype
    def reset(self, seed: int | None = None, options: dict = {}) -> Index:
        raise NotImplementedError()

    @beartype
    def render(self, mode: Literal["human", "rgb_array"] = "human"):
        if mode == "human":
            ...
        elif mode == "rgb_array":
            ...
        else:
            raise KeyError(f"Unsupported mode: {mode}")

    def close(self):
        pass
