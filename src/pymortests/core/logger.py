# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import logging

import pymor.core as core
from pymor.core.logger import log_levels
from pymor.operators.numpy import NumpyMatrixOperator
from pymortests.base import (runmodule,)


def test_logger():
    logger = NumpyMatrixOperator._logger
    for lvl in [getattr(logging, lvl) for lvl in ['WARN', 'ERROR', 'DEBUG', 'INFO']]:
        logger.setLevel(lvl)
        assert logger.isEnabledFor(lvl)
    for verb in ['warning', 'error', 'debug', 'info']:
        getattr(logger, verb)(f'{verb} -- logger {str(logger)}')


def test_empty_log_message():
    core.logger.getLogger('test').warn('')


def test_log_levels():
    logger = NumpyMatrixOperator._logger
    before_name = 'INFO'
    logger.setLevel(before_name)
    before = logger.level
    with log_levels({logger.name: 'DEBUG'}):
        assert 'DEBUG' == logging.getLevelName(logger.level)
        assert logger.level != before
    assert logger.level == before
    assert before_name == logging.getLevelName(logger.level)



if __name__ == "__main__":
    runmodule(filename=__file__)
