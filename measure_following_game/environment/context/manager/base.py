# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from os import PathLike
from pathlib import Path
from sabanamusic.common import make_score_measures, Measure
from typing import ClassVar, Literal

from measure_following_game.environment.context.manager._record import RecordBase
from measure_following_game.environment.context.renderer import ContextRenderer

__all__ = ["ContextManagerBase"]


class ContextManagerBase(object):

    metadata: ClassVar[dict] = {"render_modes": ["human", "rgb_array"]}
    num_features: ClassVar[int] = 1

    def __init__(
        self,
        score_root: str | Path | PathLike,
        record: RecordBase,
        *,
        fps: int = 20,
        onset_only: bool = True,
        window_size: int = 10,
    ):
        self.score_root = Path(score_root)
        self.fps = fps
        self.onset_only = onset_only
        self.score_measures: list[Measure] = make_score_measures(
            score_root=score_root, fps=self.fps, onset_only=self.onset_only
        )
        self.num_measures_in_score = len(self.score_measures)

        self.record = record

        if window_size > self.num_measures_in_score:
            raise ValueError("`window_size` cannot exceed number of measures in score")

        self.window_head = 0
        self.window_size = window_size
        self.window_shape = (self.window_size, self.num_features)
        self.num_measures_in_window = self.window_size

        self.similarity_matrix = np.zeros(self.window_shape, dtype=np.float32)
        self.true_measure = -1
        self.pred_measure = -1

        self.local_history: list[int] = []
        self.global_history: list[int] = []
        self.done = False

        self.renderer = ContextRenderer(score_root=self.score_root)

    @property
    def measure_range(self) -> tuple[int, int]:
        assert 0 < self.num_measures_in_window <= self.window_size
        window_tail = self.window_head + self.num_measures_in_window
        return (self.window_head, window_tail)

    @property
    def window(self) -> list[Measure]:
        window_head, window_tail = self.measure_range
        return self.score_measures[window_head:window_tail]

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

    def get_history(self, mode: Literal["global", "local"] = "local") -> list[int]:
        if mode == "global":
            return self.global_history[::]
        elif mode == "local":
            return self.local_history[::]
        else:
            raise KeyError(f"Unsupported mode: {mode}")

    def get_cursor(self, mode: Literal["global", "local"] = "local") -> int:
        if mode == "global":
            return self.global_history[-1] if self.global_history else -1
        elif mode == "local":
            return self.local_history[-1] if self.local_history else -1
        else:
            raise KeyError(f"Unsupported mode: {mode}")

    @abstractmethod
    def _fill_similarity_matrix(self):
        raise NotImplementedError()

    # TODO(kaparoo): need implementation
    def _slide_or_stay(self):
        played_measures = set(self.local_history)

    def step(self, pred_measure: int) -> tuple[np.ndarray, int, bool, dict]:
        self.pred_measure = np.clip(pred_measure, 0, self.num_measures_in_window)
        self.local_history.append(self.pred_measure)
        self.global_history.append(self.pred_measure + self.window_head)

        if self.done:
            self.true_measure = -1
            self.similarity_matrix.fill(-1.0)
        else:
            self.true_measure = self.record.true_measure - self.window_head
            self._slide_or_stay()

            # TODO(kaparoo): need implementation
            self.record.step()

            self._fill_similarity_matrix()
            self.done = self.record.done

        return self.similarity_matrix, self.true_measure, self.done, self.info

    # TODO(kaparoo): need implementation for `start_measure` and `mock`
    def _init_window_and_history(
        self, start_measure: int = 0, mock: bool = False, seed: int | None = None
    ):
        self.window_head = 0
        self.num_measures_in_window = self.window_size
        self.local_history = []
        self.global_history = []

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        start_measure, mock = 0, False
        if isinstance(options, dict):
            if "start_measure" in options:
                start_measure = options["start_measure"]
            if "mock" in options:
                mock = options["mock"]
        self._init_window_and_history(start_measure, mock, seed)

        # TODO(kaparoo): need implementation
        self.record.reset(measure_range=self.measure_range)

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
            raise KeyError(f"Unsupported mode: {mode}")
        return self.renderer.render(mode=mode)

    def close(self):
        self.renderer.close()

    def __del__(self):
        self.close()
