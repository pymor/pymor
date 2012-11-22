# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:28:14 2012

@author: r_milk01
"""

import contracts
import warnings
#import abc.

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
    For an Example see pymor.discretizer.stationary.elliptic.cg
    '''
    pass
