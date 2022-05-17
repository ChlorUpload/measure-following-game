# -*- coding: utf-8 -*-

from abc import abstractmethod
from beartype import beartype
from beartype.vale import Is
import numpy as np
from typing import Annotated, ClassVar


__all__ = ["RewardBase", "TriangleReward", "WeightedTriangleReward"]


Size = Annotated[int, Is[lambda x: x > 0]]
Index = Annotated[int, Is[lambda x: x >= 0]]
Decimal = Annotated[int | float, Is[lambda x: 0.0 <= x <= 1.0]]


class RewardBase(object):
    range: ClassVar[tuple[float, float]] = (float("-inf"), float("inf"))

    @abstractmethod
    def __call__(self, true_measure: Index, pred_measure: Index) -> float:
        raise NotImplementedError


class TriangleReward(RewardBase):
    range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    @beartype
    def __init__(self, window_size: Size):
        self.threshold = window_size // 2

    @beartype
    def __call__(self, true_measure: Index, pred_measure: Index) -> float:
        abs_error = abs(true_measure - pred_measure)
        abs_error = np.clip(abs_error, 0, self.threshold)
        return (self.threshold - abs_error) / self.threshold


class WeightedTriangleReward(TriangleReward):
    @beartype
    def __init__(
        self,
        window_size: Size,
        forward_weight: Decimal = 1.0,
        backward_weight: Decimal = 1.0,
    ):
        super().__init__(window_size)
        self.forward_weight = forward_weight
        self.backward_weight = backward_weight

    @beartype
    def __call__(self, true_measure: Index, pred_measure: Index) -> float:
        reward = super().__call__(true_measure, pred_measure)
        if true_measure <= pred_measure:
            reward *= self.forward_weight
        else:
            reward *= self.backward_weight
        return reward
