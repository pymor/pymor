# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import warnings

warn = warnings.warn

try:
    import contracts
    ContractNotRespected = contracts.ContractNotRespected
    ContractException = contracts.ContractException
except ImportError:
    pass


class ConstError(Exception):
    '''I get thrown when you try to add a new member to
    a locked class instance'''
    pass


class CallOrderWarning(UserWarning):
    '''I am raised when there's a preferred call order, but the user didn't follow it.
    '''
    pass


class AccuracyError(Exception):
    '''Is raised if the result of a computation is inaccurate'''


class ExtensionError(Exception):
    '''Is raised if a (basis) extension algorithm fails.

    This will mostly happen during a basis extension when the new snapshot is already
    in the span of the basis.
    '''


class ConfigError(Exception):
    '''Is raised if a there is any kind of problem with the keys or values in a configuration.
    '''


class InversionError(Exception):
    '''Is raised if an operator inversion algorithm fails.'''
