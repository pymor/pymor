# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Utilities for colorized log output.
via http://stackoverflow.com/questions/384076/how-can-i-make-the-python-logging-output-to-be-colored
Can not be moved because it's needed to be imported in the root __init__.py OR ELSE
"""
from __future__ import absolute_import, division, print_function
import curses
import logging
import os
import time

from pymor.core.defaults import defaults

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30
# These are the sequences needed to get colored output
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING':  YELLOW,
    'INFO':     GREEN,
    'DEBUG':    BLUE,
    'CRITICAL': MAGENTA,
    'ERROR':    RED
}

LOGLEVEL_MAPPING = {
    'debug':     logging.DEBUG,
    'info':      logging.INFO,
    'error':     logging.ERROR,
    'warn':      logging.WARN,
    'warning':   logging.WARNING,
    'critical':  logging.CRITICAL,
    'fatal':     logging.FATAL,
}

FORMAT = '%(asctime)s$BOLD%(levelname)s|$BOLD%(name)s$RESET: %(message)s'
MAX_HIERACHY_LEVEL = 3

start_time = time.time()


def formatter_message(message, use_color):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class ColoredFormatter(logging.Formatter):
    """A logging.Formatter that inserts tty control characters to color
    loglevel keyword output. Coloring can be disabled by setting the
    `PYMOR_COLORS_DISABLE` environment variable to `1`.
    """

    def __init__(self):
        disable_colors = int(os.environ.get('PYMOR_COLORS_DISABLE', 0)) == 1
        if disable_colors:
            self.use_color = False
        else:
            try:
                curses.setupterm()
                self.use_color = curses.tigetnum("colors") > 1
            except Exception:
                self.use_color = False

        super(ColoredFormatter, self).__init__(formatter_message(FORMAT, self.use_color))

    def formatTime(self, record, datefmt=None):
        elapsed = int(time.time() - start_time)
        days, remainder = divmod(elapsed, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days:
            return '{}d {:02}:{:02}:{:02}'.format(days, hours, minutes, seconds)
        elif hours:
            return '{:02}:{:02}:{:02}'.format(hours, minutes, seconds)
        else:
            return '{:02}:{:02}'.format(minutes, seconds)

    def format(self, record):
        if not record.msg:
            return ''
        tokens = record.name.split('.')
        record.name = '.'.join(tokens[1:MAX_HIERACHY_LEVEL])
        if len(tokens) > MAX_HIERACHY_LEVEL - 1:
            record.name += '.' + tokens[-1]
        levelname = record.levelname
        if self.use_color and levelname in COLORS.keys():
            if levelname is 'INFO':
                levelname_color = RESET_SEQ
            else:
                levelname_color = RESET_SEQ + '|' + COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        elif levelname is 'INFO':
            record.levelname = ''
        return logging.Formatter.format(self, record)


def getLogger(module, level=None, filename=None, handler_cls=logging.StreamHandler):
    module = 'pymor' if module == '__main__' else module
    logger = logging.getLogger(module)
    streamhandler = handler_cls()
    streamformatter = ColoredFormatter()
    streamhandler.setFormatter(streamformatter)
    logger.handlers = [streamhandler]
    logger.propagate = False
    if level:
        logger.setLevel(LOGLEVEL_MAPPING[level])
    return logger


class DummyLogger(object):

    __slots__ = []

    def nop(self, *args, **kwargs):
        return None

    propagate = False
    debug = nop
    info = nop
    warn = nop
    warning = nop
    error = nop
    critical = nop
    log = nop
    exception = nop

    def isEnabledFor(sefl, lvl):
        return False

    def getEffectiveLevel(self):
        return None

    def getChild(self):
        return self


dummy_logger = DummyLogger()


@defaults('levels')
def set_log_levels(levels={'pymor': 'WARN',
                           'pymor.core': 'WARN'}):
    for k, v in levels.items():
        getLogger(k).setLevel(v)
