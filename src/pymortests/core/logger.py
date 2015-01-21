# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import logging

import pymor.core as core
from pymortests.base import (runmodule,)

from pymortests.fixtures.generic import basicinterface_subclass


def exercise_logger(logger):
    for lvl in [getattr(logging, lvl) for lvl in ['WARN', 'ERROR', 'DEBUG', 'INFO']]:
        logger.setLevel(lvl)
        assert logger.isEnabledFor(lvl)
    for verb in ['warn', 'error', 'debug', 'info']:
        getattr(logger, verb)('{} -- logger {}'.format(verb, str(logger)))


def test_logclass(basicinterface_subclass):
    logger = basicinterface_subclass._logger
    exercise_logger(logger)


if __name__ == "__main__":
    runmodule(filename=__file__)
