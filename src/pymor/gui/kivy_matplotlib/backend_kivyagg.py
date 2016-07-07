'''
Backend KivyAgg
=====

.. image:: images/backend_agg_example.jpg
    :align: right

The :class:`FigureCanvasKivyAgg` widget is used to create a matplotlib graph.
The render will cover the whole are of the widget unless something different is
specified using a :meth:`blit`.
When you are creating a FigureCanvasKivyAgg widget, you must at least
initialize it with a matplotlib figure object. This class uses agg to get a
static image of the plot and then the image is render using a
:class:`~kivy.graphics.texture.Texture`. See backend_kivy documentation for
more information since both backends can be used in the exact same way.


Examples
--------

Example of a simple Hello world matplotlib App::

    fig, ax = plt.subplots()
    ax.text(0.6, 0.5, "hello", size=50, rotation=30.,
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                      ec=(1., 0.5, 0.5),
                      fc=(1., 0.8, 0.8),
                      )
            )
    ax.text(0.5, 0.4, "world", size=50, rotation=-30.,
            ha="right", va="top",
            bbox=dict(boxstyle="square",
                      ec=(1., 0.5, 0.5),
                      fc=(1., 0.8, 0.8),
                      )
            )
    canvas = FigureCanvasKivyAgg(figure=fig)

The object canvas can be added as a widget into the kivy tree widget.
If a change is done on the figure an update can be performed using
:meth:`~kivy.ext.mpl.backend_kivyagg.FigureCanvasKivyAgg.draw`.::

    # update graph
    canvas.draw()

The plot can be exported to png with
:meth:`~kivy.ext.mpl.backend_kivyagg.FigureCanvasKivyAgg.print_png`, as an
argument receives the `filename`.::

    # export to png
    canvas.print_png("my_plot.png")


Backend KivyAgg Events
-----------------------

The events available are the same events available from Backend Kivy.::

    def my_callback(event):
        print('press released from test', event.x, event.y, event.button)

    fig.canvas.mpl_connect('mpl_event', my_callback)

'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ('FigureCanvasKivyAgg')

import six

import matplotlib
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
    FigureManagerBase, FigureCanvasBase
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import register_backend, ShowBase

try:
    import kivy
except ImportError:
    raise ImportError("this backend requires Kivy to be installed.")

from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.base import EventLoop
from kivy.uix.floatlayout import FloatLayout
from kivy.core.image import Image
from .backend_kivy import FigureCanvasKivy,\
                            FigureManagerKivy, show, new_figure_manager,\
                            NavigationToolbar2Kivy

#register_backend('png', 'backend_kivyagg', 'PNG File Format')

toolbar = None
my_canvas = None


def new_figure_manager(num, *args, **kwargs):
    '''Create a new figure manager instance for the figure given.
    '''
    # if a main-level app must be created, this (and
    # new_figure_manager_given_figure) is the usual place to
    # do it -- see backend_wx, backend_wxagg and backend_tkagg for
    # examples. Not all GUIs require explicit instantiation of a
    # main-level app (egg backend_gtk, backend_gtkagg) for pylab
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    '''Create a new figure manager instance and a new figure canvas instance
       for the given figure.
    '''
    canvas = FigureCanvasKivyAgg(figure)
    manager = FigureManagerKivy(canvas, num)
    global my_canvas
    global toolbar
    toolbar = manager.toolbar.actionbar if manager.toolbar else None
    my_canvas = canvas
    return manager


class MPLKivyApp(App):
    '''Creates the App initializing a FloatLayout with a figure and toolbar
       widget.
    '''
    figure = ObjectProperty(None)
    toolbar = ObjectProperty(None)

    def build(self):
        EventLoop.ensure_window()
        layout = FloatLayout()
        if self.figure:
            self.figure.size_hint_y = 0.9
            layout.add_widget(self.figure)
        if self.toolbar:
            self.toolbar.size_hint_y = 0.1
            layout.add_widget(self.toolbar)
        return layout


class Show(ShowBase):
    '''mainloop needs to be overwritten to define the show() behavior for kivy
       framework.
    '''
    def mainloop(self):
        global my_canvas
        global toolbar
        app = App.get_running_app()
        if app is None:
            app = MPLKivyApp(figure=my_canvas, toolbar=toolbar)
            app.run()

show = Show()


class FigureCanvasKivyAgg(FigureCanvasKivy, FigureCanvasAgg):
    '''FigureCanvasKivyAgg class. See module documentation for more
    information.
    '''

    def __init__(self, figure, **kwargs):
        self.figure = figure
        self.bind(size=self._on_size_changed)
        super(FigureCanvasKivyAgg, self).__init__(figure=self.figure, **kwargs)
        self.img_texture = None
        self.img_rect = None
        self.blit()

    def draw(self):
        '''
        Draw the figure using the agg renderer
        '''
        self.canvas.clear()
        FigureCanvasAgg.draw(self)
        if self.blitbox is None:
            l, b, w, h = self.figure.bbox.bounds
            w, h = int(w), int(h)
            buf_rgba = self.get_renderer().buffer_rgba()
        else:
            bbox = self.blitbox
            l, b, r, t = bbox.extents
            w = int(r) - int(l)
            h = int(t) - int(b)
            t = int(b) + h
            reg = self.copy_from_bbox(bbox)
            buf_rgba = reg.to_string()
        texture = Texture.create(size=(w, h))
        texture.flip_vertical()
        with self.canvas:
            Color(1.0, 1.0, 1.0, 1.0)
            self.img_rect = Rectangle(texture=texture, pos=self.pos,
                                      size=(w, h))
        texture.blit_buffer(bytes(buf_rgba), colorfmt='rgba', bufferfmt='ubyte')
        self.img_texture = texture

    filetypes = FigureCanvasKivy.filetypes.copy()
    filetypes['png'] = 'Portable Network Graphics'

    def _on_pos_changed(self, *args):
        if self.img_rect is not None:
            self.img_rect.pos = self.pos

    def _print_image(self, filename, *args, **kwargs):
        '''Write out format png. The image is saved with the filename given.
        '''
        l, b, w, h = self.figure.bbox.bounds
        img = None
        if self.img_texture is None:
            texture = Texture.create(size=(w, h))
            texture.blit_buffer(bytes(self.get_renderer().buffer_rgba()),
                                colorfmt='rgba', bufferfmt='ubyte')
            texture.flip_vertical()
            img = Image(texture)
        else:
            img = Image(self.img_texture)
        img.save(filename)

''' Standard names that backend.__init__ is expecting '''
FigureCanvas = FigureCanvasKivyAgg
FigureManager = FigureManagerKivy
NavigationToolbar = NavigationToolbar2Kivy
show = show
