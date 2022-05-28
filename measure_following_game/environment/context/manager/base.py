# -*- coding: utf-8 -*-

__all__ = ["ContextManager"]

from abc import abstractmethod
from typing import ClassVar

from beartype import beartype
import numpy as np
from sabanamusic.models.musical import Measure, Record
from sabanamusic.common.types import Index, PositiveInt

from measure_following_game.environment.context.renderer import ContextRenderer
from measure_following_game.types import ActType, ObsType


class ContextManager(object):

    metadata: ClassVar[dict] = {"render_modes": []}
    num_features: ClassVar[PositiveInt] = 1

    @beartype
    def __init__(
        self,
        renderer: ContextRenderer,
        record: Record,
        window_size: PositiveInt = 32,
        memory_size: PositiveInt = 32,
    ):
        self.record = record

        self.renderer = renderer
        self.score_measures = renderer.score_measures
        self.num_score_measures = renderer.num_score_measures
        self.metadata["render_modes"] = renderer.modes

        if window_size > self.num_score_measures:
            raise ValueError(
                "`window_size` cannot exceed number of measures in the score"
            )

        self.window_size = window_size
        self.window_shape = (window_size, self.num_features)

        self.num_actions = window_size + 2
        self.true_action = -1
        self.pred_policy = None

        self.memory_size = memory_size
        self.memory_shape = (memory_size, self.num_actions)

        self.similarity_matrix = np.zeros(shape=self.window_shape, dtype=np.float32)
        self.memory_matrix = np.zeros(shape=self.memory_shape, dtype=np.float32)

        self.done = False

    @property
    def window_head(self) -> Index:
        return self.renderer.window_head

    @property
    def num_window_measures(self) -> PositiveInt:
        return self.renderer.num_window_measures

    @property
    def window_measures(self) -> list[Measure]:
        head = self.window_head
        tail = self.window_head + self.num_window_measures
        return self.score_measures[head:tail]

    @property
    def observation(self) -> ObsType:
        return (self.similarity_matrix, self.memory_matrix)

    @property
    @abstractmethod
    def info(self) -> dict:
        return {}

    def _init_pred_policy(self):
        self.pred_policy = np.zeros(shape=self.num_actions, dtype=np.float32)
        self.pred_policy[-1] = 1.0

    def _init_memory_matrix(self):
        self.memory_matrix.fill(0.0)
        self.memory_matrix[:, -1] = 1.0

    def _fill_memory_matrix(self):
        self.memory_matrix[:-1, :] = self.memory_matrix[1:, :]
        self.memory_matrix[-1, :] = self.pred_policy

    def _init_similarity_matrix(self):
        self.similarity_matrix.fill(0.0)

    @abstractmethod
    def _fill_similarity_matrix(self):
        raise NotImplementedError()

    @beartype
    def step(self, pred_policy: ActType) -> tuple[ObsType, int, bool, dict]:
        self.pred_policy = pred_policy

        if self.done:
            self.true_action = -1
            self._init_memory_matrix()
            self._init_similarity_matrix()
        else:
            self.true_action = self.record.true_action
            if self.true_action == -1:
                self.renderer.stay()
            elif self.true_action == -2:
                self.renderer.slide()
            else:
                self.true_action -= self.window_head
                if not (0 <= self.true_action < self.num_window_measures):
                    self.done = True
                    self.true_action = -1
                    self._init_memory_matrix()
                    self._init_similarity_matrix()
                else:
                    self.renderer.step(pred_policy)

            if not self.done:
                self.record.step()
                self._fill_memory_matrix()
                self._fill_similarity_matrix()
                self.done = self.record.done

        return self.observation, self.true_action, self.done, self.info

    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> ObsType | tuple[ObsType, dict]:
        record_options = {}
        renderer_options = {}
        if isinstance(options, dict):
            if isinstance(options.get("record"), dict):
                record_options = options["record"]
            if isinstance(options.get("renderer"), dict):
                renderer_options = options["renderer"]

        start_measure = self.renderer.reset(seed=seed, options=renderer_options)
        self.record.reset(start_measure, seed=seed, options=record_options)

        self.true_action = -1
        self._init_pred_policy()
        self._init_memory_matrix()
        self._fill_similarity_matrix()

        if return_info:
            return self.observation, self.info
        else:
            return self.observation

    @beartype
    def render(self, mode: str = "human"):
        if mode not in self.metadata["render_modes"]:
            raise KeyError(f"unsupported mode: {mode}")
        return self.renderer.render(mode=mode)

    def close(self):
        self.renderer.close()

    def __del__(self):
        self.close()
