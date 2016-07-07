import numpy as np
from os import getcwd

from pymor.gui.kivy_frontend.widgets.gl_widget import getColorBarWidget
from pymor.tools.vtkio import write_vtk
from pymor.vectorarrays.numpy import NumpyVectorArray


try:
    from kivy.uix.widget import Widget

    HAVE_KIVY = True
except ImportError:
    HAVE_KIVY = False


if HAVE_KIVY:

    def getPlotMainWindow(U, plot, length, title, isNotMatplotlib=False):

        from kivy.app import App
        from kivy.core.window import Window

        from kivy.clock import Clock

        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.gridlayout import GridLayout

        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.slider import Slider
        from kivy.uix.togglebutton import ToggleButton

        from kivy.uix.popup import Popup

        from pymor.gui.kivy_filebrowser import FileBrowser
        from kivy.utils import platform

        class PlotMainWindow(App):

            def __init__(self, U, plot, length=1, title=None, isNotMatplotlib=False):
                super(PlotMainWindow, self).__init__()

                self.title = "pyMor kivy"  # window title
                self.icon = ''  # taskbar icon
                self.title_plot = title

                self.U = U
                # find a better way to do this
                if isNotMatplotlib:
                    # gl case
                    self.plot_layout = plot.build()
                else:
                    # matplotlib case
                    self.plot_layout = plot.figure.canvas
                self.plot = plot
                self.length = length
                self.plot.set(U, 0)
                self.is_layout = isNotMatplotlib

                # this only works if the SDL window provider is used
                from kivy.base import EventLoop
                EventLoop.window.minimum_width = 800
                EventLoop.window.minimum_height = 600

            def build(self):
                #set background color
                Window.clearcolor = (1, 1, 1, 1)

                layout = BoxLayout(orientation='vertical', padding=5, spacing=5)

                title = "[b]" + self.title_plot + "[/b]"

                label = Label(text=title, markup=True, color=[0, 0, 0, 1], size_hint_y=None, height=30)

                # see http://stackoverflow.com/questions/18670687/how-i-can-adjust-variable-height-text-property-kivy

                layout.add_widget(label)
                layout.add_widget(self.plot_layout)
                self.plot.set(self.U, 0)

                if self.length > 1:
                    # time series
                    layout_slider = BoxLayout(orientation='horizontal', size_hint_y=None, height=45)

                    self.slider = Slider(min=0, max=self.length-1, size_hint_x=0.9)
                    self.slider.bind(value=self.slider_changed)

                    self.label = Label(text="0", color=[0, 0, 0, 1], size_hint_x=0.1)
                    layout_slider.add_widget(self.slider)
                    layout_slider.add_widget(self.label)
                    layout.add_widget(layout_slider)

                    layout_buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)

                    self.button_play = ToggleButton(text="Play", size_hint_x=None, width=80)
                    self.button_play.bind(on_press=self.play)

                    self.button_rewind = Button(text="Rewind", size_hint_x=None, width=80)
                    self.button_rewind.bind(on_press=self.rewind)

                    self.button_end = Button(text="End", size_hint_x=None, width=80)
                    self.button_end.bind(on_press=self.end)

                    self.button_step_back = Button(text="Step Back", size_hint_x=None, width=80)
                    self.button_step_back.bind(on_press=self.step_back)

                    self.button_step_forward = Button(text="Step Forward", size_hint_x=None, width=80)
                    self.button_step_forward.bind(on_press=self.step_forward)

                    self.button_loop = ToggleButton(text="Loop", size_hint_x=None, width=80)
                    self.button_loop.bind(on_press=self.loop)

                    label_speed = Label(text="Speed:", color=[0, 0, 0, 1], size_hint_x=None, width=70)
                    self.slider_speed = Slider(min=100, max=10000000)
                    self.slider_speed.bind(on_press=self.speed)

                    layout_buttons.add_widget(self.button_play)
                    layout_buttons.add_widget(self.button_rewind)
                    layout_buttons.add_widget(self.button_end)
                    layout_buttons.add_widget(self.button_step_back)
                    layout_buttons.add_widget(self.button_step_forward)
                    layout_buttons.add_widget(self.button_loop)

                    if self.is_layout:
                        self.button_save = Button(text="Save", size_hint_x=None, width=100)
                        self.button_save.bind(on_press=self.show_save_dialog)
                        layout_buttons.add_widget(self.button_save)

                    layout_buttons.add_widget(label_speed)
                    layout_buttons.add_widget(self.slider_speed)

                    layout.add_widget(layout_buttons)

                elif self.is_layout:
                    layout_save = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
                    self.button_save = Button(text="Save", size_hint_x=None, width=50)
                    self.button_save.bind(on_press=self.show_save_dialog)

                    layout_save.add_widget(self.button_save)

                    layout.add_widget(layout_save)

                return layout

            def start_timer(self):
                delay = self.slider_speed.value/100
                #delay = 1E-10 # for performance testing
                Clock.schedule_interval(self.update_solution, delay)

            def stop_timer(self):
                Clock.unschedule(self.update_solution)
                self.button_play.state = "normal"

            def set_solution(self, index):
                assert 0 <= index < self.length
                self.slider.value = index
                self.plot.set(self.U, index)

            def update_solution(self, dt):
                idx = self.slider.value
                if idx == self.length - 1:
                    play = self.button_play.state == "down"
                    loop = self.button_loop.state == "down"
                    # playing and looping?
                    if play and loop:
                        self.set_solution(0)
                    elif play:
                        self.button_play.state = "normal"
                        Clock.unschedule(self.update_solution)
                    return
                self.set_solution(idx+1)

            def play(self, button):
                assert button == self.button_play
                assert button.state in ["down", "normal"]
                checked = button.state == "down"
                if checked:
                    # start animation
                    self.start_timer()
                else:
                    # stop animation
                    self.stop_timer()

            def rewind(self, button):
                self.stop_timer()
                self.slider.value = 0

            def end(self, button):
                self.stop_timer()
                self.slider.value = self.length-1

            def slider_changed(self, slider, value):
                if abs(value - int(value)) > 0:
                    self.slider.value = round(value)
                    return
                assert 0 <= value < self.length
                self.label.text = str(int(value))
                self.set_solution(value)

            def step_back(self, button):
                self.stop_timer()
                idx = self.slider.value
                if idx == 0:
                    return
                self.slider.value = idx - 1

            def step_forward(self, button):
                self.stop_timer()
                idx = self.slider.value
                if idx == self.length - 1:
                    return
                self.slider.value = idx + 1

            def loop(self, button):
                pass

            def speed(self, slider, value):
                play = self.button_play.state == "down"
                if play:
                    self.stop_timer()
                    self.start_timer()

            def show_save_dialog(self, button):
                from os.path import sep, expanduser, isdir, dirname
                title = "Save as vtk file"

                # TEST
                if platform == 'win':
                    user_path = expanduser('~')
                    if not isdir(user_path + sep + 'Desktop'):
                        user_path = dirname(user_path)
                    user_path = user_path + sep + 'Documents'
                else:
                    user_path = expanduser('~') + sep + 'Documents'

                content = FileBrowser(select_string='Select',
                                      favorites=[(user_path, 'Documents')],
                                      path=getcwd())

                content.bind(on_success=self.save,
                        on_canceled=self.dismiss_popup,
                        on_submit=self.save)

                self._popup = Popup(title=title, content=content, size_hint=(0.9, 0.9))
                self._popup.open()

            def dismiss_popup(self, instance):
                self._popup.dismiss()

            def save(self, instance):
                import os
                self._popup.dismiss()

                path = instance.path
                filename = instance.filename
                filename = os.path.join(path, filename)
                base_name = filename.split('.vtu')[0].split('.vtk')[0].split('.pvd')[0]

                if base_name:
                    if len(self.U) == 1:
                        write_vtk(self.grid, NumpyVectorArray(self.U[0], copy=False), base_name, codim=self.codim)
                    else:
                        for i, u in enumerate(self.U):
                            write_vtk(self.grid, NumpyVectorArray(u, copy=False), '{}-{}'.format(base_name, i),
                                      codim=self.codim)

        return PlotMainWindow(U=U, plot=plot, length=length, title=title, isNotMatplotlib=isNotMatplotlib)

    def getPatchPlotMainWindow(U, grid, bounding_box, codim, title, legend, separate_colorbars, rescale_colorbars,
                               columns, widget, backend):

        from kivy.clock import Clock

        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        from kivy.uix.label import Label
        from kivy.uix.slider import Slider
        from kivy.uix.togglebutton import ToggleButton

        from kivy.uix.gridlayout import GridLayout
        from kivy.uix.widget import Widget

        length = len(U[0])

        # create layout
        class PlotWidget(Widget):

            PADDING = 0
            SPACING = 5

            def __init__(self):
                super(PlotWidget, self).__init__()
                if separate_colorbars:
                    if rescale_colorbars:
                        self.vmins = tuple(np.min(u[0]) for u in U)
                        self.vmaxs = tuple(np.max(u[0]) for u in U)
                    else:
                        self.vmins = tuple(np.min(u) for u in U)
                        self.vmaxs = tuple(np.max(u) for u in U)
                else:
                    if rescale_colorbars:
                        self.vmins = (min(np.min(u[0]) for u in U),) * len(U)
                        self.vmaxs = (max(np.max(u[0]) for u in U),) * len(U)
                    else:
                        self.vmins = (min(np.min(u) for u in U),) * len(U)
                        self.vmaxs = (max(np.max(u) for u in U),) * len(U)

                layout = BoxLayout(orientation='horizontal')

                plot_layout = GridLayout(cols=columns, padding=self.PADDING, spacing=self.SPACING)

                self.colorbarwidgets = [getColorBarWidget(padding=self.PADDING, vmin=vmin, vmax=vmax)
                                        for vmin, vmax in zip(self.vmins, self.vmaxs)]
                self.plots = [widget(self, grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box, codim=codim)
                         for vmin, vmax in zip(self.vmins, self.vmaxs)]

                if legend:
                    for i, plot, colorbar, l in zip(range(len(self.plots)), self.plots, self.colorbarwidgets, legend):
                        subplot_layout = BoxLayout(orientation='vertical')
                        caption = Label(text=l, color=[0, 0, 0, 1], size_hint_y=None, height=30)
                        subplot_layout.add_widget(caption)
                        if not separate_colorbars or backend == 'matplotlib':
                            subplot_layout.add_widget(plot)
                        else:
                            hlayout = BoxLayout(orientation='horizontal')
                            hlayout.add_widget(plot)
                            hlayout.add_widget(colorbar)
                            subplot_layout.add_widget(hlayout)
                        plot_layout.add_widget(subplot_layout)
                else:
                    for i, plot, colorbar in zip(range(len(self.plots)), self.plots, self.colorbarwidgets):
                        if not separate_colorbars or backend == 'matplotlib':
                            plot_layout.add_widget(plot)
                        else:
                            hlayout = BoxLayout(orientation='horizontal')
                            hlayout.add_widget(plot)
                            hlayout.add_widget(colorbar)
                            plot_layout.add_widget(hlayout)

                layout.add_widget(plot_layout)

                if not separate_colorbars:
                    layout.add_widget(self.colorbarwidgets[0])

                self.layout = layout

            def build(self):
                return self.layout

            def set(self, U, ind):
                if rescale_colorbars:
                    if separate_colorbars:
                        self.vmins = tuple(np.min(u[ind]) for u in U)
                        self.vmaxs = tuple(np.max(u[ind]) for u in U)
                    else:
                        self.vmins = (min(np.min(u[ind]) for u in U),) * len(U)
                        self.vmaxs = (max(np.max(u[ind]) for u in U),) * len(U)

                for u, plot, colorbar, vmin, vmax in zip(U, self.plots, self.colorbarwidgets, self.vmins,
                                                          self.vmaxs):
                    plot.set(u[ind], vmin=vmin, vmax=vmax)
                    colorbar.set(vmin=vmin, vmax=vmax)

        widget = PlotWidget()
        window = getPlotMainWindow(U, widget, length=length, title=title, isNotMatplotlib=True)
        window.codim = codim
        window.grid = grid
        return window
