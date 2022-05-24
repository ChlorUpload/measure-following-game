# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from os import PathLike
from pathlib import Path
from sabanamusic.common import Measure, make_score_measures
from typing import ClassVar, Literal


__all__ = ["ContextManagerBase"]


class ContextManagerBase(object):

    metadata: ClassVar[dict] = {"render_modes": []}
    num_features: ClassVar[int] = 1

    def __init__(
        self,
        score_root: str | Path | PathLike,
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

        if window_size > self.num_measures_in_score:
            raise ValueError("`window_size` cannot exceed number of measures in score")

        self.window_head = 0
        self.window_size = window_size
        self.window_shape = (self.window_size, self.num_features)
        self.num_measures_in_window = self.window_size

        self.local_history: list[int] = []
        self.global_history: list[int] = []
        self.done = False

    @property
    def window(self) -> list[Measure]:
        assert 0 < self.num_measures_in_window <= self.window_size
        window_tail = self.window_head + self.num_measures_in_window
        return self.score_measures[self.window_head : window_tail]

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

    def step(self, pred_measure: int) -> tuple[np.ndarray, int, bool, dict]:
        pred_measure = np.clip(pred_measure, 0, self.num_measures_in_window)
        self.local_history.append(pred_measure)
        self.global_history.append(pred_measure + self.window_head)
        similarity_matrix, true_measure, info = self._step_core()
        return similarity_matrix, true_measure, self.done, info

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        self.done = False
        return self._reset_core(seed=seed, return_info=return_info, options=options)

    @abstractmethod
    def _step_core(self) -> tuple[np.ndarray, int, dict]:
        ...

    @abstractmethod
    def _reset_core(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        ...

    @abstractmethod
    def render(self, mode: str):
        ...

    def close(self):
        del self.score_measures

    def __del__(self):
        self.close()
