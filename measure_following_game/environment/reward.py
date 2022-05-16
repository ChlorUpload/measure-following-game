# -*- coding: utf-8 -*-

from abc import abstractmethod


class RewardBase(object):
    @abstractmethod
    def __call__(self, true_measure: int, pred_measure: int) -> float:
        raise NotImplementedError


class TriangleReward(RewardBase):
    def __init__(self, window_size: int):
        self.threshold = window_size // 2

    def __call__(self, true_measure: int, pred_measure: int) -> float:
        abs_error = abs(true_measure - pred_measure)
        return (self.threshold - abs_error) / self.threshold
