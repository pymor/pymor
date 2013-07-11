# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Utilities for colorized log output.
via http://stackoverflow.com/questions/384076/how-can-i-make-the-python-logging-output-to-be-colored
Cannot not be moved because it's needed to be imported in the root __init__.py OR ELSE
"""
from __future__ import absolute_import, division, print_function
import logging
import curses

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30
# These are the sequences need to get colored ouput
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

FORMAT = '%(asctime)s - $BOLD%(name)s$RESET $BOLD%(levelname)s: %(message)s'
MAX_HIERACHY_LEVEL = 3


def formatter_message(message, use_color):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


class ColoredFormatter(logging.Formatter):
    """A logging.Formatter that inserts tty control characters to color
    loglevel keyword output
    """

    def __init__(self, datefmt='%H:%M:%S'):
        try:
            curses.setupterm()
            self.use_color = curses.tigetnum("colors") > 1
        except Exception, _:
            self.use_color = False
        logging.Formatter.__init__(self, formatter_message(FORMAT, self.use_color), datefmt=datefmt)

    def format(self, record):
        tokens = record.name.split('.')
        record.name = '.'.join(tokens[1:MAX_HIERACHY_LEVEL])
        if len(tokens) > MAX_HIERACHY_LEVEL - 1:
            record.name += '.' + tokens[-1]
        levelname = record.levelname
        if self.use_color and levelname in COLORS.keys():
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def getLogger(module, level=None, filename=None):
    module = 'pymor' if module == '__main__' else module
    logger_name = module
    logger = logging.getLogger(module)
    streamhandler = logging.StreamHandler()
    streamformatter = ColoredFormatter()
    streamhandler.setFormatter(streamformatter)
    logger.handlers = [streamhandler]
    logger.propagate = False
    if level:
        logger.setLevel(LOGLEVEL_MAPPING[level])
    return logger

dummy_logger = getLogger('pymor.dummylogger', level='fatal')
