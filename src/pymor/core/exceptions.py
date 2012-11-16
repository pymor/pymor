# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:28:14 2012

@author: r_milk01
"""

import contracts

ContractNotRespected = contracts.ContractNotRespected

class ConstError(Exception):
    '''I get thrown when you try to add a new member to
    a locked class instance'''
    pass

class CodimError(Exception):
    '''Is raised if an invalid codimension index is unsed.'''
    pass
