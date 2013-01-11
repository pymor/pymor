"""Utilities for colorized log output.
via http://stackoverflow.com/questions/384076/how-can-i-make-the-python-logging-output-to-be-colored
Cannot not be moved because it's needed to be imported in the root __init__.py OR ELSE  
"""

import logging
from buildbot.util.bbcollections import defaultdict

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30
#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING': YELLOW,
    'INFO': GREEN,
    'DEBUG': BLUE,
    'CRITICAL': MAGENTA,
    'ERROR': RED
}

LOGLEVEL_MAPPING = {
            'debug'    : logging.DEBUG,
            'info'     : logging.INFO,
            'error'    : logging.ERROR,
            'warn'     : logging.WARN,
            'warning'  : logging.WARNING,
            'critical' : logging.CRITICAL,
            'fatal'    : logging.FATAL,
        }

FORMAT = '$BOLD%(levelname)s $BOLD%(name)s$RESET %(asctime)s - %(message)s'
MAX_HIERACHY_LEVEL = 2

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

class ColoredFormatter(logging.Formatter):
    """A logging.Formatter that inserts tty control characters to color
    loglevel keyword output
    """

    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

def getLogger(module, level='debug', filename=None):
    module = 'pymor' if module == '__main__' else module
    logger_name = '.'.join(module.split('.')[:MAX_HIERACHY_LEVEL])
    logger = logging.getLogger(logger_name)
    streamhandler = logging.StreamHandler()
    streamformatter = ColoredFormatter(formatter_message(FORMAT, True))
    streamhandler.setFormatter(streamformatter)
    logger.handlers = [streamhandler]
    logger.propagate = False
    return logger


