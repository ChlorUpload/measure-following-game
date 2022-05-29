# -*- coding: utf-8 -*-

from pathlib import Path
import unittest

from gym import spaces
import numpy as np

from measure_following_game.environment import *
from measure_following_game.params import *
from measure_following_game.utils import *


class EnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        score_root = Path(__file__).parents[2] / "samples"
        record_name = "record/demo"
        cls.env_param = make_env_param(score_root=score_root, record_name=record_name)

    def test_runs(self):
        make_env(self.env_param)

    def test_spaces(self):
        env = make_env(self.env_param)

        action_space = env.action_space
        self.assertIsInstance(action_space, spaces.Box)
        self.assertEqual(action_space.shape, (self.env_param.window_size,))
        self.assertEqual(action_space.dtype, np.float32)

        observation_space = env.observation_space
        self.assertIsInstance(observation_space, spaces.Tuple)
        self.assertEqual(len(observation_space), 2)

        similarity_matrix = observation_space[0]
        self.assertIsInstance(similarity_matrix, spaces.Box)
        self.assertEqual(similarity_matrix.shape, env.manager.window_shape)
        self.assertEqual(similarity_matrix.dtype, np.float32)

        memory_matrix = observation_space[1]
        self.assertIsInstance(memory_matrix, spaces.Box)
        self.assertEqual(memory_matrix.shape, env.manager.memory_shape)
        self.assertEqual(memory_matrix.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
