# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import logging

import pytest

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
    core.logger.getLogger('test').warning('')


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


@pytest.mark.parametrize('verb', ('info', 'error', 'fatal', 'debug', 'block', 'info2', 'info3', 'warning'))
def test_once(verb, capsys):
    logger = NumpyMatrixOperator._logger
    logger.setLevel('DEBUG')
    func = getattr(logger, f'{verb}_once')
    msg = f'{verb} -- logger {str(logger)}'
    func(msg)
    # this just clears the capture buffer
    capsys.readouterr()
    func(msg)
    second = capsys.readouterr()
    # same log call must result in no output
    assert second.out == second.err == ''


if __name__ == "__main__":
    runmodule(filename=__file__)
