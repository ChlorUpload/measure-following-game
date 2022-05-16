# -*- coding: utf-8 -*-

import abc
from typing import ClassVar


class RewardBase(object):
    range: ClassVar[tuple[float, float]] = (float("-inf"), float("inf"))

    @abc.abstractmethod
    def __call__(self, true_measure: int, pred_measure: int) -> float:
        raise NotImplementedError


class TriangleReward(RewardBase):
    range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    def __init__(self, window_size: int):
        self.threshold = window_size // 2

    def __call__(self, true_measure: int, pred_measure: int) -> float:
        abs_error = abs(true_measure - pred_measure)
        return (self.threshold - abs_error) / self.threshold
