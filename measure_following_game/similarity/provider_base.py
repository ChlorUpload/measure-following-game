# -*- coding: utf-8 -*-

import abc
import numpy as np


__all__ = ["SimilarityProviderBase"]


class SimilarityProviderBase(object):
    def __init__(self):
        ...

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
