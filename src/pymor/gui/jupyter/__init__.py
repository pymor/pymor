# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides plotting support inside the Jupyter notebook.

To use these routines you first have to execute ::

        %matplotlib notebook

inside the given notebook.
"""
import IPython

from pymor.core import logger
from pymor.core.logger import ColoredFormatter

from ipywidgets import IntProgress, HTML, VBox
import ipywidgets
import logging


def progress_bar(sequence, every=None, size=None, name='Parameters'):
    # c&p from https://github.com/kuk/log-progress
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    IPython.display.display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = f"{name}: {str(index or '?')}"


class LogViewer(logging.Handler):
    out = None

    def __init__(self, out, accordion=None):
        super().__init__()
        self.out = out
        self.accordion = accordion
        self.setFormatter(ColoredFormatter())
        self.first_emit = True

    def emit(self, record):
        if self.first_emit:
            if self.accordion:
                IPython.display.display(self.accordion)
            self.first_emit = False
        record = self.formatter.format_html(record)
        self.out.value += f'<p style="line-height:120%">{record}</p>'

    @property
    def empty(self):
        return len(self.out.value) == 0

    def close(self):
        if self.empty and self.accordion:
            self.accordion.close()

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.out)


class LoggingRedirector(object):

    def __init__(self):
        self.old_handlers = None
        self.old_default = None
        self.new_handler = None
        self.accordion = None

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        out = ipywidgets.HTML(layout=ipywidgets.Layout(width='100%', height='16em', overflow_y='auto'))

        self.accordion = ipywidgets.widgets.Accordion(children=[out])
        self.accordion.set_title(0, 'Log Output')
        # start collapsed
        self.accordion.selected_index = None

        self.new_handler = LogViewer(out, self.accordion)

        def _new_default(_):
            return [self.new_handler]

        self.old_default = logger.default_handler
        logger.default_handler = _new_default
        self.old_handlers = {name: logging.getLogger(name).handlers for name in logging.root.manager.loggerDict}
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).handlers = [self.new_handler]

    def stop(self):
        if self.old_default is None:
            # %load_ext in the frist cell triggers a post_run_cell with no matching pre_run_cell event before
            return
        self.new_handler.close()
        logger.default_handler = self.old_default
        for name in logging.root.manager.loggerDict:
            try:
                logging.getLogger(name).handlers = self.old_handlers[name]
            except KeyError:
                # loggers that have been created during the redirect get a default handler
                logging.getLogger(name).handlers = logger.default_handler()


redirect_logging = LoggingRedirector()


def load_ipython_extension(ipython):
    ipython.events.register('pre_run_cell', redirect_logging.start)
    ipython.events.register('post_run_cell', redirect_logging.stop)


def unload_ipython_extension(ipython):
    ipython.events.unregister('pre_run_cell', redirect_logging.start)
    ipython.events.unregister('post_run_cell', redirect_logging.stop)