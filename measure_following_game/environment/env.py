# -*- coding: utf-8 -*-

__all__ = ["MeasureFollowingEnv"]

from beartype import beartype
from gym import Env, spaces

from measure_following_game.types import ActType, ObsType
from measure_following_game.environment.context import ContextManager
from measure_following_game.environment.rewards import Reward


class MeasureFollowingEnv(Env[ObsType, ActType]):
    @beartype
    def __init__(self, manager: ContextManager, reward: Reward):
        self.manager = manager
        self.metadata.update(manager.metadata)
        self.reward = reward
        self.reward_range = reward.range
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(manager.window_size,))
        self.observation_space = spaces.Tuple(
            spaces.Box(low=0.0, high=1.0, shape=manager.window_shape),
            spaces.Box(low=0.0, high=1.0, shape=manager.memory_shape),
        )

    @beartype
    def step(self, pred_policy: ActType) -> tuple[ObsType, float, bool, dict]:
        assert self.action_space.contains(pred_policy)
        observation, true_action, done, info = self.manager.step(pred_policy)
        reward = self.reward(true_action, pred_policy)
        return observation, reward, done, info

    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None
    ) -> ObsType | tuple[ObsType, dict]:
        return self.manager.reset(seed=seed, return_info=return_info, options=options)

    @beartype
    def render(self, mode: str = "human"):
        return self.manager.render(mode)

    def close(self):
        self.manager.close()

    def __del__(self):
        self.close()
