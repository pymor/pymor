# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from time import perf_counter

import numpy as np

from pymor.core.config import config

config.require('IPYWIDGETS')

from IPython.display import display
from ipywidgets import Accordion, Button, Checkbox, FloatSlider, HBox, Label, Layout, Text, VBox, jsdlink

from pymor.analyticalproblems.functions import ExpressionFunction
from pymor.core.base import BasicObject
from pymor.parameters.base import ParameterSpace


class ParameterSelector(BasicObject):
    """Parameter selector."""

    def __init__(self, space, dim_input):
        assert isinstance(space, ParameterSpace)
        self.space = space
        self._handlers = []

        class WidgetBase:
            def __init__(self):
                self._old_values = [w.value for w in self._widgets]
                self._handlers = []
                for obj in self._widgets:
                    obj.observe(self._values_changed, 'value')
                    obj._valid_value = True
                self.valid = True

            def _call_handlers(self):
                for handler in self._handlers:
                    handler()

            def _values_changed(self, change):
                was_valid = self.valid
                # do nothing if new values are invalid
                if not all(t._valid_value for t in self._widgets):
                    self.valid = False
                    if was_valid:
                        self._call_handlers()
                    return

                self.valid = True
                new_values = [t.value for t in self._widgets]
                if new_values != self._old_values:
                    self._old_values = new_values
                    self._call_handlers()

            @property
            def values(self):
                return self._old_values

            def on_change(self, handler):
                self._handlers.append(handler)

        class ParameterWidget(WidgetBase):
            def __init__(self, p):
                dim = space.parameters[p]
                low, high = space.ranges[p]
                self._widgets = sliders = []
                for i in range(dim):
                    sliders.append(FloatSlider((high+low)/2, min=low, max=high,
                                               description=f'{i}:'))
                self.widget = Accordion(titles=[p], children=[VBox(sliders)], selected_index=0)
                super().__init__()

        class InputWidget(WidgetBase):
            def __init__(self, dim):
                self._widgets = texts = []
                for i in range(dim):
                    texts.append(Text('0.', description=f'{i}:'))

                    def text_changed(change):
                        try:
                            input = ExpressionFunction(change['new'], dim_domain=1, variable='t')
                            if input.shape_range not in [(), (1,)]:
                                raise ValueError
                            change['owner'].style.background = '#FFFFFF'
                            change['owner']._valid_value = True
                            return
                        except ValueError:
                            change['owner'].style.background = '#FFCCCC'
                            change['owner']._valid_value = False

                    texts[i]._valid_input = True
                    texts[i].observe(text_changed, 'value')

                self.widget = Accordion(titles=['input'], children=[VBox(texts)], selected_index=0)
                super().__init__()

        self._widgets = _widgets = {p: ParameterWidget(p) for p in space.parameters}
        if dim_input > 0:
            assert 'input' not in _widgets
            _widgets['input'] = InputWidget(dim_input)
        for w in _widgets.values():
            w.on_change(self._update_mu)
        self._auto_update = Checkbox(value=False, indent=False, description='auto update',
                                     layout=Layout(flex='0'))
        self._update_button = Button(description='Update', disabled=False)
        self._update_button.on_click(self._on_update)
        jsdlink((self._auto_update, 'value'), (self._update_button, 'disabled'))
        controls = HBox([self._auto_update, self._update_button],
                        layout=Layout(border='solid 1px lightgray',
                                      margin='2px',
                                      padding='2px',
                                      justify_content='space-around'))
        self.widget = VBox([w.widget for w in _widgets.values()] + [controls])
        self._update_mu()
        self.last_mu = self.mu

    def display(self):
        return display(self.widget)

    def on_change(self, handler):
        self._handlers.append(handler)

    def _call_handlers(self):
        self._update_button.disabled = True
        for handler in self._handlers:
            handler(self.mu, self.input)
        self.last_mu, self.last_input = self.mu, self.input

    def _update_mu(self):
        if any(not w.valid for w in self._widgets.values()):
            self._update_button.disabled = True
            return
        self.mu = self.space.parameters.parse({p: w.values for p, w in self._widgets.items() if p != 'input'})
        if 'input' in self._widgets:
            self.input = ExpressionFunction('[' + ','.join(self._widgets['input'].values) + ']',
                                            dim_domain=1, variable='t')
        else:
            self.input = None
        if self._auto_update.value:
            self._call_handlers()
        else:
            self._update_button.disabled = False

    def _on_update(self, b):
        self._call_handlers()


def interact(model, parameter_space, show_solution=True, visualizer=None, transform=None):
    """Interactively explore |Model| in jupyter environment.

    This method dynamically creates a set of `ipywidgets` to interactively visualize
    a model's solution and output.

    Parameters
    ----------
    model
        The |Model| to interact with.
    parameter_space
        |ParameterSpace| within which the |Parameters| of the model can be chosen.
    show_solution
        If `True`, show the model's solution for the given parameters.
    visualizer
        A method of the form `visualize(U, return_widget=True)` which is called to obtain
        an `ipywidget` that renders the solution. If `None`, `model.visualize` is used.
    transform
        A method `transform(U, mu)` returning the data that is passed to the `visualizer`.
        If `None` the solution `U` is passed directly.

    Returns
    -------
    The created widgets as a single `ipywidget`.
    """
    assert model.parameters == parameter_space.parameters
    right_pane = []
    parameter_selector = ParameterSelector(parameter_space, model.dim_input)
    right_pane.append(parameter_selector.widget)

    has_output = model.dim_output > 0
    tic = perf_counter()
    mu, input = parameter_selector.mu, parameter_selector.input
    data = model.compute(solution=show_solution, output=has_output, mu=mu, input=input)
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
            fig.canvas.layout.flex = '1 0 320px'
            fig.set_figwidth(320 / 100)
            fig.set_figheight(200 / 100)
            output_lines = ax.plot(output)
            fig.legend([str(i) for i in range(model.dim_output)])
            output_widget = fig.canvas
        else:
            labels = [Text(str(o), description=f'{i}:', disabled=True) for i, o in enumerate(output.ravel())]
            output_widget = VBox(labels)
        right_pane.append(Accordion(titles=['output'], children=[output_widget], selected_index=0))

    sim_time_widget = Label(f'{sim_time}s')
    right_pane.append(HBox([Label('simulation time:'), sim_time_widget]))

    right_pane = VBox(right_pane)

    if show_solution:
        U = data['solution']
        if transform:
            U = transform(U, mu)
        visualizer = (visualizer or model.visualize)(U, return_widget=True)
        visualizer.layout.flex = '0.6 0 auto'
        right_pane.layout.flex = '0.4 1 auto'
        widget = HBox([visualizer, right_pane])
        widget.layout.grid_gap = '5%'
    else:
        widget = right_pane

    def do_update(mu, input):
        tic = perf_counter()
        data = model.compute(solution=show_solution, output=has_output, mu=mu, input=input)
        sim_time = perf_counter() - tic
        if show_solution:
            U = data['solution']
            if transform:
                U = transform(U, mu)
            visualizer.set(U)
        if has_output:
            output = data['output']
            if len(output) > 1:
                for l, o in zip(output_lines, output.T):
                    l.set_ydata(o)
                low, high = ax.get_ylim()
                ax.set_ylim(min(low, np.min(output)), max(high, np.max(output)))
                output_widget.draw_idle()
            else:
                for l, o in zip(output_widget.children, output.ravel()):
                    l.value = str(o)
        sim_time_widget.value = f'{sim_time}s'

    parameter_selector.on_change(do_update)

    return widget
