# -*- coding: utf-8 -*-

import unittest
import random

from beartype.roar import *
import numpy as np

from measure_following_game.environment.rewards import *


class RewardTest(unittest.TestCase):
    def test_init(self):
        window_size = random.randint(1, 100)
        reward = Reward(window_size=window_size)

        self.assertIsInstance(reward, Reward)
        self.assertEqual(reward.window_size, window_size)
        self.assertEqual(reward.num_actions, window_size + 2)

        with self.assertRaises(BeartypeCallHintParamViolation):
            Reward(window_size=random.randint(-100, 0))

        with self.assertRaises(BeartypeCallHintParamViolation):
            Reward(window_size=random.uniform(-100.0, 100.0))

    def test_call(self):
        reward = Reward(window_size=10)  # num_actions = 12

        with self.assertRaises(BeartypeCallHintParamViolation):
            reward(1, 1)

        with self.assertRaises(AssertionError):
            reward(1, np.zeros(12, dtype=np.float32))

        with self.assertRaises(AssertionError):
            reward(1, np.ones(12, dtype=np.float32))

        with self.assertRaises(BeartypeCallHintParamViolation):
            pred_actions = np.zeros(12, dtype=np.int32)
            pred_actions[3] = 1
            reward(1, pred_actions)

        with self.assertRaises(NotImplementedError):
            reward(1, np.random.dirichlet([1] * 12).astype(np.float32))


class TriangleRewardTest(unittest.TestCase):
    def test_init(self):
        window_size = random.randint(20, 40)
        reward = TriangleReward(window_size=window_size)

        self.assertIsInstance(reward, TriangleReward)
        self.assertEqual(reward.threshold, window_size // 2)

    def test_deterministic_call(self):
        window_size = 7  # num_actions = 9, treshold = 3
        reward = TriangleReward(window_size=window_size)

        true_action = 1

        # pred_actions_1: move cursor to measure 0
        pred_actions_1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.assertEqual(reward(true_action, pred_actions_1), 2 / 3)

        # pred_actions_2: move cursor to measure 5
        # since reward.threshold is 3, abs_error is 3, not 4 due to clipping
        pred_actions_2 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
        self.assertEqual(reward(true_action, pred_actions_2), 0.0)

        # pred_actions_3: slide window
        pred_actions_3 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
        self.assertEqual(reward(true_action, pred_actions_3), 0.0)


if __name__ == "__main__":
    unittest.main()
