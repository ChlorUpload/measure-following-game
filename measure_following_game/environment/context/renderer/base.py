# -*- coding: utf-8 -*-

__all__ = ["ContextRenderer"]

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from beartype import beartype
from sabanamusic.common import make_score_measures, Measure
from sabanamusic.typing import Index, PathLike, PositiveInt

from measure_following_game.types import ActType


class ContextRenderer(object):

    modes: ClassVar[list[str]] = ["human", "rgb_array"]

    @beartype
    def __init__(
        self,
        score_root: PathLike,
        fps: PositiveInt = 20,
        onset_only: bool = True,
    ):
        self.score_root = Path(score_root)
        self.fps, self.onset_only = fps, onset_only
        self.score_measures: list[Measure] = make_score_measures(
            score_root=score_root, fps=fps, onset_only=onset_only
        )
        self.num_score_measures = len(self.score_measures)

        self.window_head: Index = 0
        self.num_window_measures: PositiveInt = 0

    @abstractmethod
    def stay(self):
        ...

    @abstractmethod
    def slide(self):
        ...

    @abstractmethod
    def step(self, pred_policy: ActType):
        ...

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict = {}) -> Index:
        ...

    @abstractmethod
    def render(self, mode: str):
        ...

    @abstractmethod
    def close(self):
        ...

    def __del__(self):
        self.close()
