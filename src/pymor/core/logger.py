# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module contains pyMOR's logging facilities.

pyMOR's logging facilities are based on the :mod:`logging` module of the
Python standard library. To obtain a new logger object use :func:`getLogger`.
Logging can be configured via the :func:`set_log_format` and
:func:`set_log_levels` methods.
"""

import logging
import os
import time
from contextlib import contextmanager
from functools import lru_cache
from types import MethodType

from pymor.core.defaults import defaults
from pymor.tools import mpi

BLOCK = logging.INFO + 5
BLOCK_TIME = BLOCK + 1
INFO2 = logging.INFO + 1
INFO3 = logging.INFO + 2
logging.addLevelName(BLOCK, 'BLOCK')
logging.addLevelName(BLOCK_TIME, 'BLOCK_TIME')
logging.addLevelName(INFO2, 'INFO2')
logging.addLevelName(INFO3, 'INFO3')

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30
# These are the sequences needed to get colored output
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING':  YELLOW,
    'INFO2':    YELLOW,
    'INFO3':    RED,
    'DEBUG':    BLUE,
    'CRITICAL': MAGENTA,
    'ERROR':    RED
}
CSSCOLORS = {
    'WARNING':  'yellow',
    'INFO2':    'yellow',
    'INFO3':    'red',
    'DEBUG':    'blue',
    'CRITICAL': 'magenta',
    'ERROR':    'red'
}

MAX_HIERARCHY_LEVEL = 1
BLOCK_TIMINGS = True
INDENT_BLOCKS = True
INDENT = 0
LAST_TIMESTAMP_LENGTH = 0

start_time = time.perf_counter()


class ColoredFormatter(logging.Formatter):
    """A logging.Formatter that colors loglevel keyword output.

    Coloring can be disabled by setting the `PYMOR_COLORS_DISABLE` environment variable to `1`.
    """

    def __init__(self):
        disable_colors = int(os.environ.get('PYMOR_COLORS_DISABLE', 0)) == 1
        if disable_colors:
            self.use_color = False
        else:
            try:
                import curses
                curses.setupterm()
                self.use_color = curses.tigetnum("colors") > 1
            except Exception:
                self.use_color = False

        super().__init__()

    def _format_common(self, record):
        global LAST_TIMESTAMP_LENGTH

        msg = super().format(record)  # call base class to support exception formatting

        # format time
        elapsed = int(time.perf_counter() - start_time)
        days, remainder = divmod(elapsed, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        timestamp = f'{days}d {hours:02}:{minutes:02}:{seconds:02}' if days \
            else f'{hours:02}:{minutes:02}:{seconds:02}' if hours \
            else f'{minutes:02}:{seconds:02}'
        if not mpi.rank0:
            timestamp = f'RANK{mpi.rank}|{timestamp}'
        if LAST_TIMESTAMP_LENGTH == 0:
            LAST_TIMESTAMP_LENGTH = len(timestamp)

        # handle special cases
        if not record.msg:
            return ' ' * (LAST_TIMESTAMP_LENGTH+1) + '|   ' * INDENT
        if record.levelname == 'BLOCK_TIME':
            return ' ' * (LAST_TIMESTAMP_LENGTH+1) + '|   ' * (INDENT - 1) + r'\----------------- ' + record.msg

        # handle length change of timestamp
        if len(timestamp) > LAST_TIMESTAMP_LENGTH:
            timestep_length = len(timestamp)
            if INDENT > 0:
                for i in reversed(range(LAST_TIMESTAMP_LENGTH, timestep_length)):
                    timestamp = ' ' * (i + 2) + r'\   ' * INDENT + '\n' + timestamp
            LAST_TIMESTAMP_LENGTH = timestep_length

        indent = '|   ' * INDENT

        tokens = record.name.split('.')
        if len(tokens) > MAX_HIERARCHY_LEVEL - 1:
            path = '.'.join(tokens[1:MAX_HIERARCHY_LEVEL] + [tokens[-1]])
        else:
            path = '.'.join(tokens[1:MAX_HIERARCHY_LEVEL])

        levelname = record.levelname
        return levelname, path, msg, timestamp, indent

    def format(self, record):
        try:
            ret = self._format_common(record)
            levelname, path, msg, timestamp, indent = ret
        except ValueError:
            return ret

        if self.use_color:
            if levelname in ('INFO', 'BLOCK'):
                path = BOLD_SEQ + path + RESET_SEQ
                levelname = ''
            elif levelname.startswith('INFO'):
                path = (COLOR_SEQ % (30 + COLORS[levelname])) + path + RESET_SEQ
                levelname = ''
            else:
                path = BOLD_SEQ + path + RESET_SEQ
                levelname = (COLOR_SEQ % (30 + COLORS[levelname])) + '|' + levelname + '|' + RESET_SEQ
        else:
            if levelname in ('INFO', 'BLOCK'):
                levelname = ''
            else:
                levelname = '|' + levelname + '|'

        return f'{timestamp} {indent}{levelname}{path}: {msg}'

    def format_html(self, record):
        try:
            ret = self._format_common(record)
            levelname, path, msg, timestamp, indent = ret
        except ValueError:
            return ret

        levelname, path, msg, timestamp, indent = self._format_common(record)

        if levelname in ('INFO', 'BLOCK'):
            path = f'<bold>{path}</bold>'
            levelname = ''
        elif levelname.startswith('INFO'):
            path = f'<span style="color:{CSSCOLORS[levelname]};">{path}</span>'
            levelname = ''
        else:
            path = f'<bold>{path}</bold>'
            levelname = f'<span style="color:{CSSCOLORS[levelname]};">|{levelname}|</span>'

        return f'{timestamp} {indent}{levelname}{path}: {msg}'


@defaults('filename')
def default_handler(filename=None):
    streamhandler = logging.StreamHandler()
    streamformatter = ColoredFormatter()
    streamhandler.setFormatter(streamformatter)
    handlers = [streamhandler]
    if filename:
        filehandler = logging.FileHandler(filename)
        fileformatter = ColoredFormatter()
        filehandler.setFormatter(fileformatter)
        handlers.append(filehandler)
    return handlers


@defaults('filename')
def getLogger(module, level=None, filename=None):
    """Get the logger of the respective module for pyMOR's logging facility.

    In addition to the logging methods inherited from :class:`~logging.Logger`
    all returned loggers get a block, info2, info3 method for the
    respective new levels. Plus all warnings methods get a `XXX_once` method
    that caches the msg and only emits the log entry the first time (per logger
    instance, not globally).

    Parameters
    ----------
    module
        Name of the module.
    level
        If set, `logger.setLevel(level)` is called (see
        :meth:`~logging.Logger.setLevel`).
    filename
        If not empty, path of an existing file where everything logged will be
        written to.
    """
    module = 'pymor' if module == '__main__' else module
    logger = logging.getLogger(module)
    logger.block = MethodType(_block, logger)
    logger.info2 = MethodType(_info2, logger)
    logger.info3 = MethodType(_info3, logger)
    for level_function in ('info', 'error', 'fatal', 'debug', 'block', 'info2', 'info3', 'warning'):
        # add a method that is wrapped in a cache, so calls with same args aren't executed again
        setattr(logger, f'{level_function}_once', lru_cache(None)(getattr(logger, level_function)))
    logger.handlers = default_handler(filename)
    logger.propagate = False
    if level:
        logger.setLevel(level)
    return logger


class DummyLogger:

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

    def block(self, msg, *args, **kwargs):
        return LogIndenter(self, False)

    def info2(self, msg, *args, **kwargs):
        self.log(INFO2, msg, *args, **kwargs)

    def info3(self, msg, *args, **kwargs):
        self.log(INFO3, msg, *args, **kwargs)


dummy_logger = DummyLogger()


@defaults('levels')
def set_log_levels(levels=None):
    """Set log levels for pyMOR's logging facility.

    Parameters
    ----------
    levels
        Dict of log levels. Keys are names of loggers (see :func:`logging.getLogger`),
        values are the log levels to set for the loggers of the given names
        (see :meth:`~logging.Logger.setLevel`).
    """
    levels = levels or {'pymor': 'INFO'}
    for k, v in levels.items():
        getLogger(k).setLevel(v)


@defaults('max_hierarchy_level', 'indent_blocks', 'block_timings')
def set_log_format(max_hierarchy_level=1, indent_blocks=True, block_timings=False):
    """Set log levels for pyMOR's logging facility.

    Parameters
    ----------
    max_hierarchy_level
        The number of components of the loggers name which are printed.
        (The first component is always stripped, the last component always
        preserved.)
    indent_blocks
        If `True`, indent log messages inside a code block started with
        `with logger.block(...)`.
    block_timings
        If `True`, measure the duration of a code block started with
        `with logger.block(...)`.
    """
    global MAX_HIERARCHY_LEVEL
    global INDENT_BLOCKS
    global BLOCK_TIMINGS
    MAX_HIERARCHY_LEVEL = max_hierarchy_level
    INDENT_BLOCKS = indent_blocks
    BLOCK_TIMINGS = block_timings


class LogIndenter:

    def __init__(self, logger, doit):
        self.logger = logger
        self.doit = doit

    def __enter__(self):
        global INDENT
        global BLOCK_TIMINGS
        if BLOCK_TIMINGS:
            self.tic = time.perf_counter()
        if self.doit:
            INDENT += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        global INDENT
        global BLOCK_TIMINGS
        if self.doit:
            if BLOCK_TIMINGS:
                duration = time.perf_counter() - self.tic
                self.logger.log(BLOCK_TIME, f'duration: {duration}s')
            INDENT -= 1


def _block(self, msg, *args, **kwargs):
    global INDENT_BLOCKS
    self.log(BLOCK, msg, *args, **kwargs)
    return LogIndenter(self, self.isEnabledFor(BLOCK) and INDENT_BLOCKS)


def _info2(self, msg, *args, **kwargs):
    self.log(INFO2, msg, *args, **kwargs)


def _info3(self, msg, *args, **kwargs):
    self.log(INFO3, msg, *args, **kwargs)


@contextmanager
def log_levels(level_mapping):
    """Change levels for given loggers on entry and reset to before state on exit.

    Parameters
    ----------
    level_mapping
        a dict of logger name -> level name
    """
    for name, level in level_mapping.items():
        logger = getLogger(name)
        level_mapping[name] = logger.level
        logger.setLevel(level)
    try:
        yield
    finally:
        for name, level in level_mapping.items():
            getLogger(name).setLevel(level)
