# -*- coding: utf-8 -*-

from gym import spaces
from measure_following_game.environment import *
import numpy as np
from pathlib import Path
import unittest


class EnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        score_root = Path(__file__).parent / "samples"
        provider = ContextManagerBase(score_root, window_size=10)
        reward = RewardBase()
        cls.env = MeasureFollowingEnv(provider, reward)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_spaces(self):
        action_space = self.env.action_space
        self.assertIsInstance(action_space, spaces.Discrete)
        self.assertEqual(action_space.n, self.env.window_shape[0])
        self.assertEqual(action_space.dtype, np.int64)

        observation_space = self.env.observation_space
        self.assertIsInstance(observation_space, spaces.Box)
        self.assertEqual(observation_space.shape, self.env.window_shape)
        self.assertEqual(observation_space.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
