'''
Created on Nov 19, 2012

@author: r_milk01
'''

class CallOrderWarning(UserWarning):
    '''I am raised when there's a preferred call order, but the user didn't follow it.
    For an Example see pymor.discretizer.stationary.elliptic.cg
    '''
    pass