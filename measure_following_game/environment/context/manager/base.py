# -*- coding: utf-8 -*-

from abc import abstractmethod
from beartype import beartype
import numpy as np
from pathlib import Path
from sabanamusic.common import make_score_measures, Measure, RecordBase
from sabanamusic.typing import Index, PathLike, PositiveInt
from typing import ClassVar, Literal

from measure_following_game.environment.context.renderer import ContextRenderer


__all__ = ["ContextManagerBase"]


class ContextManagerBase(object):

    metadata: ClassVar[dict] = {"render_modes": ["human", "rgb_array"]}
    num_features: ClassVar[PositiveInt] = 1

    @beartype
    def __init__(
        self,
        score_root: PathLike,
        record: RecordBase,
        renderer: ContextRenderer,
        *,
        fps: PositiveInt = 20,
        onset_only: bool = True,
        window_size: PositiveInt = 32,
    ):
        self.score_root = Path(score_root)
        self.fps, self.onset_only = fps, onset_only
        self.score_measures: list[Measure] = make_score_measures(
            score_root=self.score_root, fps=self.fps, onset_only=self.onset_only
        )
        self.num_measures_in_score = len(self.score_measures)

        self.record = record
        self.renderer = renderer

        if window_size > self.num_measures_in_score:
            raise ValueError("`window_size` cannot exceed number of measures in score")

        self.window_head = 0
        self.window_size = window_size
        self.num_measures_in_window = self.window_size

        self.similarity_matrix = np.zeros(self.window_shape, dtype=np.float32)
        self.true_measure = -1
        self.pred_measure = -1

        self.local_history: list[Index] = []
        self.global_history: list[Index] = []
        self.done = False

    @property
    def measure_range(self) -> tuple[Index, Index]:
        return (self.window_head, self.window_head + self.num_measures_in_window)

    @property
    def window(self) -> list[Measure]:
        window_head, window_tail = self.measure_range
        return self.score_measures[window_head:window_tail]

    @property
    def window_shape(self) -> tuple[PositiveInt, PositiveInt]:
        return (self.window_size, self.num_features)

    @property
    def info(self) -> dict:
        return {
            "measure_range": self.measure_range,
            "cursor": {
                "global": self.get_cursor(mode="global"),
                "local": self.get_cursor(mode="local"),
            },
            "history": {
                "global": self.get_history(mode="global"),
                "local": self.get_history(mode="local"),
            },
        }

    def get_history(self, mode: Literal["global", "local"] = "global") -> list[Index]:
        if mode not in ("global", "local"):
            raise KeyError("Unsupported mode %s" % mode)
        history = self.global_history if mode == "global" else self.local_history
        return history[::]

    def get_cursor(
        self, mode: Literal["global", "local"] = "global"
    ) -> Index | Literal[-1]:
        if mode not in ("global", "local"):
            raise KeyError("Unsupported mode %s" % mode)
        history = self.global_history if mode == "global" else self.local_history
        return history[-1] if history else -1

    @abstractmethod
    def _fill_similarity_matrix(self):
        raise NotImplementedError()

    # TODO(kaparoo): need implementation
    def _slide_or_stay(self):
        self.local_history  # do something when slide
        self.window_head = 0
        self.num_measures_in_window = self.window_size

    @beartype
    def step(
        self, pred_measure: Index
    ) -> tuple[np.ndarray, Index | Literal[-1], bool, dict]:
        num_measures_in_window = self.num_measures_in_window
        self.pred_measure = np.clip(pred_measure, 0, num_measures_in_window - 1)
        self.local_history.append(self.pred_measure)
        self.global_history.append(self.pred_measure + self.window_head)

        if self.done:
            self.true_measure = -1
            self.similarity_matrix.fill(0.0)
        else:
            self.true_measure = self.record.true_measure - self.window_head
            if not (0 <= self.true_measure < num_measures_in_window):
                self.true_measure = -1
                self.similarity_matrix.fill(0.0)
                self.done = True
            else:
                self._slide_or_stay()
                self.record.step()
                self._fill_similarity_matrix()
                self.done = self.record.done

        return self.similarity_matrix, self.true_measure, self.done, self.info

    # TODO(kaparoo): need implementation
    def _init_window_and_history(self, start_measure: int = 0):
        self.local_history = []
        self.global_history = []
        self.window_head = 0
        self.num_measures_in_window = self.window_size

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        start_measure = 0
        if isinstance(options, dict) and "start_measure" in options:
            start_measure = int(options["start_measure"])
        self._init_window_and_history(start_measure)
        self.record.reset(start_measure, seed=seed)
        self._fill_similarity_matrix()
        self.done = self.record.done

        if return_info:
            return self.similarity_matrix, self.info
        else:
            return self.similarity_matrix

    def render(
        self, mode: Literal["human", "rgb_array"] = "human"
    ) -> np.ndarray | None:
        if mode not in ("human", "rgb_array"):
            raise KeyError("Unsupported mode: %s" % mode)
        # TODO(kaparoo): need implementation
        return self.renderer.render(mode=mode)

    def close(self):
        self.renderer.close()

    def __del__(self):
        self.close()
