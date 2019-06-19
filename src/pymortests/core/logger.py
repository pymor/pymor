# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import logging

import pymor.core as core
from pymortests.base import (runmodule,)

from pymortests.fixtures.generic import basicinterface_subclass


def exercise_logger(logger):
    for lvl in [getattr(logging, lvl) for lvl in ['WARN', 'ERROR', 'DEBUG', 'INFO']]:
        logger.setLevel(lvl)
        assert logger.isEnabledFor(lvl)
    for verb in ['warning', 'error', 'debug', 'info']:
        getattr(logger, verb)(f'{verb} -- logger {str(logger)}')


def test_logclass(basicinterface_subclass):
    logger = basicinterface_subclass._logger
    exercise_logger(logger)


def test_empty_log_message():
    core.logger.getLogger('test').warn('')


if __name__ == "__main__":
    runmodule(filename=__file__)
