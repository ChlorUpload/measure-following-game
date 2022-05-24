# -*- coding: utf-8 -*-

from beartype.roar import *
from measure_following_game.environment.reward import *
import unittest


class RewardTest(unittest.TestCase):
    def test_init_type(self):
        WeightedTriangleReward(window_size=1, forward_weight=1, backward_weight=1)

        with self.assertRaises(BeartypeCallHintParamViolation):
            WeightedTriangleReward(window_size=1.0)

    def test_init_value(self):
        with self.assertRaises(BeartypeCallHintParamViolation):
            WeightedTriangleReward(window_size=0)

        with self.assertRaises(BeartypeCallHintParamViolation):
            WeightedTriangleReward(window_size=-1)

        with self.assertRaises(BeartypeCallHintParamViolation):
            WeightedTriangleReward(window_size=1, forward_weight=10)

        with self.assertRaises(BeartypeCallHintParamViolation):
            WeightedTriangleReward(window_size=1, backward_weight=-1)

    def test_call_type(self):
        reward = WeightedTriangleReward(
            window_size=10, forward_weight=1.0, backward_weight=0.5
        )

        reward(true_measure=2, pred_measure=4)

        with self.assertRaises(BeartypeCallHintParamViolation):
            reward(true_measure=-1, pred_measure=4)

        with self.assertRaises(BeartypeCallHintParamViolation):
            reward(true_measure=2, pred_measure=4.0)

    def test_call_value(self):
        reward = WeightedTriangleReward(
            window_size=10, forward_weight=1.0, backward_weight=0.5
        )

        self.assertEqual(reward(true_measure=2, pred_measure=4), 0.6)
        self.assertEqual(reward(true_measure=2, pred_measure=7), 0.0)
        self.assertEqual(reward(true_measure=2, pred_measure=8), 0.0)

        self.assertEqual(reward(true_measure=2, pred_measure=0), 0.3)
        self.assertEqual(reward(true_measure=2, pred_measure=1), 0.4)


if __name__ == "__main__":
    unittest.main()
