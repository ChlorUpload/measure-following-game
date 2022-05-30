# -*- coding: utf-8 -*-

__all__ = ["ContextRenderer"]

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from beartype import beartype
from numpy import random
from sabanamusic.common.types import Index, PathLike, PositiveInt
from sabanamusic.models.musical import make_score_measures, Measure
from sabanamusic.models.graphical import Sheet, SheetView, JsonIO

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

        self.json_io = JsonIO()
        self._init_sheet_view(kwargs.get("layout_name"))

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

    def _init_sheet_view(self, layout_name: str | None = None):
        if layout_name is not None and not layout_name.strip():
            json_path = (self.score_root / layout_name).with_suffix(".json")
            sheet = self.json_io.parse(json_path)
        else:
            sheet = Sheet.create_random_sheet(
                self.num_score_measures,
                measure_width_prob_distribution=[0.3, 0.6, 0.1],
                staff_wide_prob=0.25,
                layout_width=random.choice(list(range(800, 1600))),
            )

        self.sheet_view = SheetView(sheet)

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    def __del__(self):
        self.close()
