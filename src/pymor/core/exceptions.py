# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:28:14 2012

@author: r_milk01
"""

class ConstError(Exception):
    '''I get thrown when you try to add a new member to
    a locked class instance'''
    pass