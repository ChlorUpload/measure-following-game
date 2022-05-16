# -*- coding: utf-8 -*-

import gym
from measure_following_game.similarity import SimilarityProviderBase
from .reward import RewardBase
import numpy as np


__all__ = ["MeasureFollowingEnv"]


class MeasureFollowingEnv(gym.Env[np.ndarray, int]):
    def __init__(self, reward: RewardBase, provider: SimilarityProviderBase):
        self.reward = reward
        self.provider = provider

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        similarity_matrix, true_measure, done = self.provider.step(action)
        reward = self.reward(true_measure, action)
        return similarity_matrix, reward, done, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        super(MeasureFollowingEnv, self).reset(seed=seed)
        return self.provider.reset(seed=seed, return_info=return_info, options=options)

    def render(self, mode: str):
        self.provider.render(mode)

    def close(self):
        self.provider.close()
