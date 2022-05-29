# -*- coding: utf-8 -*-

__all__ = ["ContextRenderer"]

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from beartype import beartype
from sabanamusic.common.types import Index, PathLike, PositiveInt
from sabanamusic.models.musical import make_score_measures, Measure

from measure_following_game.types import ActType


class ContextRenderer(object):

    render_modes: ClassVar[list[str]] = ["human", "rgb_array"]

    @beartype
    def __init__(
        self,
        score_root: PathLike,
        fps: PositiveInt = 20,
        onset_only: bool = True,
        **kwargs,
    ):
        self.score_root = Path(score_root)
        self.fps, self.onset_only = fps, onset_only
        self.score_measures: list[Measure] = make_score_measures(
            score_root=score_root, fps=fps, onset_only=onset_only
        )
        self.num_score_measures = len(self.score_measures)

    @property
    def window_head(self) -> Index:
        raise NotImplementedError()

    @property
    def num_window_measures(self) -> PositiveInt:
        raise NotImplementedError()

    def stay(self):
        pass

    @abstractmethod
    def slide(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, pred_policy: ActType):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict = {}) -> Index:
        raise NotImplementedError()

    @abstractmethod
    def render(self, mode: str = "human"):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    def __del__(self):
        self.close()
