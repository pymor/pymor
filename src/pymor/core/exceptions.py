# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import contracts
import warnings

warn = warnings.warn
ContractNotRespected = contracts.ContractNotRespected
ContractException = contracts.ContractException


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
