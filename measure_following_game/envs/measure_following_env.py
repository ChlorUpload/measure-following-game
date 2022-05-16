# -*- coding: utf-8 -*-

import gym
import numpy as np

from measure_following_game.similarity import SimilarityProviderBase


__all__ = ["MeasureFollowingEnv"]


class MeasureFollowingEnv(gym.Env[np.ndarray, int]):
    def __init__(self, provider: SimilarityProviderBase):
        self.provider = provider

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        similarity_matrix, true_measure, is_done = self.provider.step(action)
        ...

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
