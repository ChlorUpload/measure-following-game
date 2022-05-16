# -*- coding: utf-8 -*-

import abc
import numpy as np
from typing import ClassVar

__all__ = ["SimilarityProviderBase"]


class SimilarityProviderBase(object):

    metadata: ClassVar[dict] = {"render_modes": []}
    num_features: ClassVar[int] = 1

    def __init__(self, window_size: int):
        self.window_offset = 0
        self.window_size = window_size

    @abc.abstractmethod
    def step(self, pred_measure: int) -> tuple[np.ndarray, int, bool]:
        ...

    @abc.abstractmethod
    def reset(
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
