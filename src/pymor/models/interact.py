# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from time import perf_counter

import numpy as np

from pymor.core.config import config

config.require('IPYWIDGETS')

from IPython.display import display
from ipywidgets import Button, Checkbox, FloatSlider, HBox, Label, VBox, jsdlink

from pymor.core.base import BasicObject
from pymor.parameters.base import Mu, ParameterSpace


class ParameterSelector(BasicObject):

    def __init__(self, space):
        assert isinstance(space, ParameterSpace)
        self.space = space
        self._handlers = []

        class ParameterWidget:
            def __init__(self, p):
                dim = space.parameters[p]
                low, high = space.ranges[p]
                widgets = []
                for i in range(dim):
                    widgets.append(FloatSlider((high-low)/2, min=low, max=high))
                self._widgets = widgets
                self.widget = HBox([Label(f'{p}:'), VBox(widgets)])

            @property
            def values(self):
                return [w.value for w in self._widgets]

            def on_change(self, handler):
                for w in self._widgets:
                    w.observe(lambda change: handler(), 'value')

        self._widgets = _widgets = {p: ParameterWidget(p) for p in space.parameters}
        for w in _widgets.values():
            w.on_change(self._update_mu)
        self._auto_update = Checkbox(value=False, description='auto update')
        self._update_button = Button(description='Update', disabled=False)
        self._update_button.on_click(self._on_update)
        jsdlink((self._auto_update, 'value'), (self._update_button, 'disabled'))
        self.widget = VBox([w.widget for w in _widgets.values()] +
                           [HBox([self._auto_update, self._update_button])])
        self._update_mu()
        self.last_mu = self.mu


    def display(self):
        return display(self.widget)

    def on_change(self, handler):
        self._handlers.append(handler)

    def _call_handlers(self):
        for handler in self._handlers:
            handler(self.mu)
        self.last_mu = self.mu

    def _update_mu(self):
        self.mu = Mu({p: w.values for p, w in self._widgets.items()})
        if self._auto_update.value:
            self._call_handlers()
        else:
            self._update_button.disabled = False

    def _on_update(self, b):
        b.disabled = True
        self._call_handlers()


def interact(model, parameter_space, show_solution=True, visualizer=None):
    right_pane = []
    parameter_selector = ParameterSelector(parameter_space)
    right_pane.append(parameter_selector.widget)

    has_output = model.dim_output > 0
    tic = perf_counter()
    data = model.compute(solution=show_solution, output=has_output, mu=parameter_selector.mu)
    sim_time = perf_counter() - tic

    if has_output:
        output = data['output']
        if len(output) > 1:
            from IPython import get_ipython
            from matplotlib import pyplot as plt
            get_ipython().run_line_magic('matplotlib', 'widget')
            plt.ioff()
            fig, ax = plt.subplots(1,1)
            fig.canvas.header_visible = False
            output_lines = ax.plot(output)
            fig.legend([str(i) for i in range(model.dim_output)])
            output_widget = fig.canvas
        else:
            output_widget = Label(str(output.ravel()))
        right_pane.append(HBox([Label('output:'), output_widget]))

    sim_time_widget = Label(f'{sim_time}s')
    right_pane.append(HBox([Label('simulation time:'), sim_time_widget]))

    right_pane = VBox(right_pane)

    if show_solution:
        U = data['solution']
        visualizer = visualizer(U) if visualizer is not None else model.visualize(U, return_widget=True)
        widget = HBox([visualizer, right_pane])
    else:
        widget = right_pane

    def do_update(mu):
        tic = perf_counter()
        data = model.compute(solution=show_solution, output=has_output, mu=mu)
        sim_time = perf_counter() - tic
        if show_solution:
            visualizer.set(data['solution'])
        if has_output:
            output = data['output']
            if len(output) > 1:
                for l, o in zip(output_lines, output.T):
                    l.set_ydata(o)
                low, high = ax.get_ylim()
                ax.set_ylim(min(low, np.min(output)), max(high, np.max(output)))
                output_widget.draw_idle()
            else:
                output_widget.value = str(output.ravel())
        sim_time_widget.value = f'{sim_time}s'

    parameter_selector.on_change(do_update)

    return widget
