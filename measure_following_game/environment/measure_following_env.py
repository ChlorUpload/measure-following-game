# -*- coding: utf-8 -*-

import gym
import gym.spaces as spaces
import numpy as np

from measure_following_game.environment.context import ContextManagerBase
from measure_following_game.environment.reward import RewardBase


__all__ = ["MeasureFollowingEnv"]


class MeasureFollowingEnv(gym.Env[np.ndarray, int]):
    def __init__(self, manager: ContextManagerBase, reward: RewardBase):
        self.manager = manager
        self.window_shape = self.manager.window_shape
        self.action_space = spaces.Discrete(n=self.window_shape[0])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.window_shape)
        self.metadata |= self.manager.metadata
        self.reward = reward
        self.reward_range = self.reward.range

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.action_space.contains(action)
        similarity_matrix, true_measure, done, info = self.manager.step(action)
        reward = self.reward(true_measure, action)
        return similarity_matrix, reward, done, info

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        super(MeasureFollowingEnv, self).reset(seed=seed)  # reset self._np_random
        return self.manager.reset(seed=seed, return_info=return_info, options=options)

    def render(self, mode: str):
        self.manager.render(mode)

    def close(self):
        self.manager.close()

    def __del__(self):
        self.close()
