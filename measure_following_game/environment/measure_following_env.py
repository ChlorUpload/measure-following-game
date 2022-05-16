# -*- coding: utf-8 -*-

import gym
from gym import spaces
import numpy as np

from .reward import RewardBase
from .similarity.provider import SimilarityProviderBase


__all__ = ["MeasureFollowingEnv"]


class MeasureFollowingEnv(gym.Env[np.ndarray, int]):
    def __init__(self, reward: RewardBase, provider: SimilarityProviderBase):
        self.reward = reward
        self.reward_range = self.reward.range
        self.provider = provider
        window_size = self.provider.window_size
        num_features = self.provider.num_features
        self.action_space = spaces.Discrete(window_size)
        self.observation_space = spaces.Box(0.0, 1.0, (window_size, num_features))
        self.metadata["render_modes"] = self.provider.metadata["render_modes"]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action)
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
        assert mode in self.metadata["render_modes"]
        self.provider.render(mode)

    def close(self):
        self.provider.close()
