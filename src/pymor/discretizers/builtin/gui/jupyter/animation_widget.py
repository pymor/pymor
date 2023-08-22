# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('IPYWIDGETS')

from ipywidgets import HBox, IntSlider, Layout, Play, jslink


class AnimationWidget(HBox):
    """A nicer animation widget than the builtin Play ipywidget."""

    def __init__(self, frames):
        self.play = Play(min=None, max=frames-1)
        self.frame_slider = IntSlider(0, 0, frames-1, layout=Layout(flex='1'))
        self.speed_slider = IntSlider(value=100, min=10, max=1000, description='delay:', readout=False,
                                      layout=Layout(flex='0.5'))
        jslink((self.play, 'value'), (self.frame_slider, 'value'))
        jslink((self.speed_slider, 'value'), (self.play, 'interval'))
        super().__init__([self.play, self.frame_slider, self.speed_slider])
