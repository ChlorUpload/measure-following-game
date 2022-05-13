# -*- coding: utf-8 -*-

import gym
import numpy as np


class MeasureFollowingEnv(gym.Env[np.ndarray, int]):
    def __init__(self):
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        ...

    def render(self, mode):
        ...

    def close(self):
        ...
