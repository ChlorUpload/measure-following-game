# -*- coding: utf-8 -*-

import os
import pathlib
import unittest

from measure_following_game.params import *
from measure_following_game.utils import *


class ParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file_path = pathlib.Path(__file__).parent / "params.json"

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.file_path)

    def test_env_param(self):
        param1 = make_env_param(score_root="somewhere", record_name="unknown")
        param1.save_json(self.file_path)
        param2 = EnvParam.load_json(self.file_path)
        self.assertEqual(param1.config, param2.config)


if __name__ == "__main__":
    unittest.main()
