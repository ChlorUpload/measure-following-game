# -*- coding: utf-8 -*-

__all__ = ["GridContextRenderer"]

from argparse import ArgumentError
import os
import re
from typing import ClassVar, Literal

from beartype import beartype
import numpy as np
import pygame
from sabanamusic.common.types import Index, PathLike, PositiveInt
from sabanamusic.models.graphical import Measure, Sheet, SheetView, JsonIO

from measure_following_game.environment.context.renderer.base import ContextRenderer
from measure_following_game.types import ActType


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PLAIN = (234, 247, 241)
ACTIVATE = (204, 227, 201)

image_regex = r"score_img_([0-9]+)\.png"


class GridContextRenderer(ContextRenderer):

    render_modes: ClassVar[list[str]] = ["human", "rgb_array"]

    @beartype
    def __init__(
        self,
        score_root: PathLike,
        fps: PositiveInt = 20,
        onset_only: bool = True,
        **kwargs,
    ):
        super().__init__(score_root, fps, onset_only, **kwargs)

        self.sheet_height = 0
        for staff in self.sheet_view.sheet.staves:
            self.sheet_height += staff.height

        self.screen_width = self.sheet_view.sheet.layout_width
        self.screen_height = self.sheet_view.display_height

        self.measure_rects: list = []

        self.cursor = 0
        self.scroll_top = 0

        self.channel_last = kwargs.get("channel_last") is True

        self.measure_images = {}

        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.surf = pygame.Surface((self.screen_width, self.screen_height))

        self.surf.fill(WHITE)
        self._calc_resized_measure_rects()
        self._load_measure_images()
        self._render_layout_initial_measures()

    @property
    def visible_measures(self) -> list[Measure]:
        return self.sheet_view.get_visible_measures()

    @property
    def visible_indices(self) -> list[Index]:
        return [m.index for m in self.visible_measures]

    @property
    def window_head(self) -> Index:
        return self.visible_measures[0].index

    @property
    def num_window_measures(self) -> PositiveInt:
        return len(self.visible_measures)

    def slide(self):
        self.sheet_view.slide()
        scroll_dest = self.measure_rects[self.window_head][1]
        # TODO(kaparoo): need smooth scroll
        self.scroll_top = scroll_dest

    @beartype
    def step(self, pred_policy: ActType):
        index = np.argmax(pred_policy)
        if index in self.visible_indices:
            self.cursor = index

    @beartype
    def reset(self, seed: int | None = None, options: dict = {}) -> Index:
        self._init_sheet_view(options.get("layout_name"))
        self.scroll_top = 0
        self.cursor = 0
        if isinstance(start_staff_idx := options.get("start_staff_idx"), int):
            sheet_view = self.sheet_view
            for _ in range(len(sheet_view.sheet.staves)):
                if start_staff_idx == sheet_view.get_visible_staves()[0].index:
                    break
                else:
                    sheet_view.slide()
            else:
                raise ArgumentError()
        np.random.seed(seed)
        start_measure: Measure = np.random.choice(self.visible_measures)
        return start_measure.index

    @beartype
    def render(self, mode: Literal["human", "rgb_array"] = "human"):
        self._render_layout_visible_measures()
        self.screen.fill(WHITE)
        self.screen.blit(
            self.surf,
            (0, 0),
            area=[0, self.scroll_top, self.screen_width, self.screen_height],
        )

        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            rgb_array = pygame.surfarray.pixels3d(self.screen)  # W X H X C
            channel_order = (1, 0, 2) if self.channel_last else (2, 1, 0)
            return np.transpose(rgb_array, channel_order)
        else:
            raise KeyError(f"Unsupported mode: {mode}")

    def _calc_resized_measure_rects(self):
        self.measure_rects = []
        cur_x = 0
        cur_y = 0
        for staff in self.sheet_view.sheet.staves:
            total_measure_width = 0
            for measure in staff.measures:
                total_measure_width += measure.width

            weight = self.screen_width / total_measure_width
            for measure in staff.measures:
                resized_width = measure.width * weight
                measure_rect = (cur_x, cur_y, resized_width, staff.height)
                self.measure_rects.append(measure_rect)
                cur_x += resized_width

            cur_x = 0
            cur_y += staff.height

    def _load_measure_images(self):
        image_dir = self.score_root / "score_images"
        if not os.path.exists(image_dir):
            return

        for image_path in image_dir.glob("score_img_*.png"):
            image_path = str(image_path)
            res = re.findall(image_regex, image_path)
            if not res or len(res) != 1:
                break
            index = int(res[0])

            image = pygame.image.load(image_path)
            _, _, rw, rh = self.measure_rects[index]
            iw = image.get_width()
            ih = image.get_height()

            ratio_w = rw / iw
            ratio_h = rh / ih
            ratio = min(ratio_w, ratio_h)

            image = pygame.transform.smoothscale(image, (iw * ratio, ih * ratio))
            self.measure_images[index] = image

    def _render_layout_initial_measures(self):
        self.surf.fill(WHITE)
        self._render_measures(lambda _: PLAIN)

    def _render_layout_visible_measures(self):
        def callback(index):
            if index in self.visible_indices:
                return ACTIVATE if self.cursor == index else PLAIN
            else:
                return None

        self._render_measures(callback)

    def _render_measures(self, get_color_by_index):
        for index, rect in enumerate(self.measure_rects):
            color = get_color_by_index(index)
            if color:
                pygame.draw.rect(self.surf, color, rect)
                pygame.draw.rect(self.surf, BLACK, rect, 1)
                if index in self.measure_images:
                    image = self.measure_images.get(index)
                    iw = image.get_width()
                    ih = image.get_height()
                    x, y, w, h = rect
                    self.surf.blit(image, (x + w / 2 - iw / 2, y + h / 2 - ih / 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
