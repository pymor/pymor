from __future__ import absolute_import, division, print_function

import pymor.core as core
from pymortests.base import (runmodule,)

from pymortests.fixtures import basicinterface_subclass   # NOQA


def exercise_logger(logger):
    for lvl in core.logger.LOGLEVEL_MAPPING.values():
        logger.setLevel(lvl)
        assert logger.isEnabledFor(lvl)
    for verb in ['warn', 'error', 'debug', 'info']:
        getattr(logger, verb)('{} -- logger {}'.format(verb, str(logger)))


def test_logclass(basicinterface_subclass):
    logger = basicinterface_subclass._logger
    exercise_logger(logger)


if __name__ == "__main__":
    runmodule(filename=__file__)
