# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import contracts
import warnings

warn = warnings.warn
ContractNotRespected = contracts.ContractNotRespected
ContractException = contracts.ContractException

class CommunicationError(Exception):
    '''Is raised when the `data` field of a `Communicable`
    is accessed, but communication is disabled.
    '''

class ConstError(Exception):
    '''I get thrown when you try to add a new member to
    a locked class instance'''
    pass


class CodimError(Exception):
    '''Is raised if an invalid codimension index is used.'''
    pass


class CallOrderWarning(UserWarning):
    '''I am raised when there's a preferred call order, but the user didn't follow it.
    For an Example see pymor.discretizers.stationary.elliptic.cg
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
