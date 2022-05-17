# -*- coding: utf-8 -*-

import abc
import numpy as np
from typing import ClassVar, Literal

__all__ = ["SimilarityProviderBase"]


class SimilarityProviderBase(object):

    metadata: ClassVar[dict] = {"render_modes": []}
    num_features: ClassVar[int] = 1

    def __init__(self, window_size: int = 10):
        self.window_head = 0
        self.window_size = window_size
        self.num_measures = window_size
        self.local_context: list[int] = []
        self.global_context: list[int] = []

    def get_context(self, mode: Literal["local", "global"] = "local") -> list[int]:
        return self.local_context[::] if mode == "local" else self.global_context[::]

    def get_cursor(self, mode: Literal["local", "global"] = "local") -> int:
        return self.local_context[-1] if mode == "local" else self.global_context[-1]

    @abc.abstractmethod
    def _slide_or_stay(self):
        ...

    def step(self, pred_measure: int) -> tuple[np.ndarray, int, bool, dict]:
        pred_measure = np.clip(pred_measure, 0, self.num_measures - 1)
        self.local_context.append(pred_measure)
        self.global_context.append(pred_measure + self.window_head)
        self._slide_or_stay()
        similarity_matrix, true_measure, done, info = self._step_core()
        info |= {"local_context": self.get_context(), "local_cursor": self.get_cursor()}
        return similarity_matrix, true_measure, done, info

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        self.window_head = 0
        self.num_measures = self.window_size
        self.local_context = []
        self.global_context = []
        return self._reset_core(seed=seed, return_info=return_info, options=options)

    @abc.abstractmethod
    def _step_core(self) -> tuple[np.ndarray, int, bool, dict]:
        ...

    @abc.abstractmethod
    def _reset_core(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        ...

    @abc.abstractmethod
    def render(self, mode: str):
        ...

    @abc.abstractmethod
    def close(self):
        ...

    def __del__(self):
        self.close()
