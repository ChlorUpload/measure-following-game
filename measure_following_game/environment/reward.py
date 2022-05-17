# -*- coding: utf-8 -*-

import abc
import numpy as np
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
        abs_error = np.clip(abs_error, 0, self.threshold)
        return (self.threshold - abs_error) / self.threshold


class WeightedTraingleReward(TriangleReward):
    def __init__(
        self,
        window_size: int,
        forward_weight: float = 1.0,
        backward_weight: float = 1.0,
    ):
        super().__init__(window_size)
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight

    def __call__(self, true_measure: int, pred_measure: int) -> float:
        reward = super().__call__(true_measure, pred_measure)
        if true_measure <= pred_measure:
            reward *= self.forward_weight
        else:
            reward *= self.backward_weight
        return reward
