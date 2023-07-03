# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain
from time import perf_counter

import numpy as np

from pymor.core.config import config

config.require('IPYWIDGETS')

from IPython.display import display
from ipywidgets import Accordion, Button, Checkbox, FloatSlider, HBox, Label, Layout, Stack, Text, VBox, jsdlink

from pymor.core.base import BasicObject
from pymor.models.basic import StationaryModel
from pymor.parameters.base import Mu, Parameters, ParameterSpace


class ParameterSelector(BasicObject):
    """Parameter selector."""

    def __init__(self, space, time_dependent):
        assert isinstance(space, ParameterSpace)
        self.space = space
        self._handlers = []

        class ParameterWidget:
            def __init__(self, p):
                dim = space.parameters[p]
                low, high = space.ranges[p]
                self._sliders = sliders = []
                self._texts = texts = []
                self._checkboxes = checkboxes = []
                stacks = []
                hboxes = []
                for i in range(dim):
                    sliders.append(FloatSlider((high+low)/2, min=low, max=high,
                                               description=f'{i}:'))

                for i in range(dim):
                    texts.append(Text(f'{(high+low)/2:.2f}', description=f'{i}:'))

                    def text_changed(change):
                        try:
                            Parameters(p=1).parse(change['new'])
                            change['owner'].style.background = '#FFFFFF'
                            return
                        except ValueError:
                            pass
                        change['owner'].style.background = '#FFCCCC'

                    texts[i].observe(text_changed, 'value')

                    stacks.append(Stack(children=[sliders[i], texts[i]], selected_index=0, layout=Layout(flex='1')))
                    checkboxes.append(Checkbox(value=False, description='time dep.',
                                               indent=False, layout=Layout(flex='0')))

                    def check_box_clicked(change, idx):
                        stacks[idx].selected_index = int(change['new'])

                    checkboxes[i].observe(lambda change, idx=i: check_box_clicked(change, idx), 'value')

                    hboxes.append(HBox([stacks[i], checkboxes[i]]))
                widgets = VBox(hboxes if time_dependent else sliders)
                self.widget = Accordion(titles=[p], children=[widgets], selected_index=0)
                self._old_values = [s.value for s in sliders]
                self.valid = True
                self._handlers = []

                for obj in chain(sliders, texts, checkboxes):
                    obj.observe(self._values_changed, 'value')

            def _call_handlers(self):
                for handler in self._handlers:
                    handler()

            def _values_changed(self, change):
                was_valid = self.valid
                new_values = [t.value if c.value else s.value
                              for s, t, c in zip(self._sliders, self._texts, self._checkboxes)]
                # do nothing if new values are invalid
                try:
                    Parameters(p=len(self._sliders)).parse(new_values)
                except ValueError:
                    self.valid = False
                    if was_valid:
                        self._call_handlers()
                    return

                self.valid = True
                if new_values != self._old_values:
                    self._old_values = new_values
                    self._call_handlers()

            @property
            def values(self):
                return self._old_values

            def on_change(self, handler):
                self._handlers.append(handler)

        self._widgets = _widgets = {p: ParameterWidget(p) for p in space.parameters}
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
            handler(self.mu)
        self.last_mu = self.mu

    def _update_mu(self):
        if any(not w.valid for w in self._widgets.values()):
            self._update_button.disabled = True
            return
        self.mu = self.space.parameters.parse({p: w.values for p, w in self._widgets.items()})
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
    if model.dim_input > 0:
        params = Parameters(model.parameters, input=model.dim_input)
        parameter_space = ParameterSpace(params, dict(parameter_space.ranges, input=[-1,1]))
    right_pane = []
    parameter_selector = ParameterSelector(parameter_space, time_dependent=not isinstance(model, StationaryModel))
    right_pane.append(parameter_selector.widget)

    has_output = model.dim_output > 0
    tic = perf_counter()
    mu = parameter_selector.mu
    input = parameter_selector.mu.get('input', None)
    mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
            for k in mu if k != 'input'})
    data = model.compute(solution=show_solution, output=has_output, input=input, mu=mu)
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

    def do_update(mu):
        if 'input' in mu:
            input = mu.get_time_dependent_value('input') if mu.is_time_dependent('input') else mu['input']
        else:
            input = None
        mu = Mu({k: mu.get_time_dependent_value(k) if mu.is_time_dependent(k) else mu[k]
                for k in mu if k != 'input'})
        tic = perf_counter()
        data = model.compute(solution=show_solution, output=has_output, input=input, mu=mu)
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
