# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('IPYTHON')
config.require('IPYWIDGETS')


import logging

import ipywidgets
from IPython.display import display

from pymor.core import logger
from pymor.core.logger import ColoredFormatter


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
                display(self.accordion)
            self.first_emit = False
        record = self.formatter.format_html(record)
        self.out.value += f'<div style="font-family:monospace,monospace;line-height:120%">{record}<br></div>'

    @property
    def empty(self):
        return len(self.out.value) == 0

    def close(self):
        if self.empty and self.accordion:
            self.accordion.close()

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.out)


class LoggingRedirector:

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
            # %load_ext in the first cell triggers a post_run_cell
            # with no matching pre_run_cell event before
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

# AFAICT there is no robust way to query for loaded extensions
# and we have to make sure we do not setup two redirects
_extension_loaded = False

def load_ipython_extension(ipython):
    global _extension_loaded
    ipython.events.register('pre_run_cell', redirect_logging.start)
    ipython.events.register('post_run_cell', redirect_logging.stop)
    _extension_loaded = True


def unload_ipython_extension(ipython):
    global _extension_loaded
    ipython.events.unregister('pre_run_cell', redirect_logging.start)
    ipython.events.unregister('post_run_cell', redirect_logging.stop)
    _extension_loaded = False
