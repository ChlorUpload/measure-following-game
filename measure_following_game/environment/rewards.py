# -*- coding: utf-8 -*-

__all__ = ["Reward", "TriangleReward"]

from abc import abstractmethod
from typing import ClassVar

from beartype import beartype
import numpy as np
import numpy.typing as npt
from sabanamusic.typing import PositiveInt

from measure_following_game.types import ActType


class Reward(object):

    range: ClassVar[tuple[float, float]] = (float("-inf"), float("+inf"))

    @beartype
    def __init__(self, window_size: PositiveInt = 32):
        self.window_size = window_size
        self.num_actions = window_size + 2  # +2 for `slide` and `noop`

    @beartype
    def __call__(self, true_action: int, pred_policy: ActType) -> float:
        assert len(pred_policy) == self.num_actions
        expected_reward = 0.0
        for pred_action, pred_prob in enumerate(pred_policy):
            expected_reward += pred_prob * self._calc_reward(true_action, pred_action)
        return expected_reward

    def _is_measure(self, action: int) -> bool:
        return 0 <= action < self.num_actions

    @abstractmethod
    def _calc_reward(self, true_action: int, pred_action: int) -> float:
        raise NotImplementedError()


class TriangleReward(Reward):

    range: ClassVar[tuple[float, float]] = (0.0, float("+inf"))

    @beartype
    def __init__(self, window_size: PositiveInt = 32):
        super().__init__(window_size)
        self.threshold = window_size // 2

    def _calc_reward(self, true_action: int, pred_action: int) -> float:
        if self._is_measure(true_action) and self._is_measure(pred_action):
            abs_error = abs(true_action - pred_action)
            abs_error = np.clip(abs_error, 0.0, self.threshold)
            return (self.threshold - abs_error) / self.threshold
        return 0.0
