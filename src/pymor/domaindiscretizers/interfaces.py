from __future__ import absolute_import, division, print_function

import pymor.core as core


class DomainDiscretizerInterface(core.BasicInterface):
    '''Takes DomainDiscription and generates AffineGrid and BoundaryInfo'''

    @core.interfaces.abstractmethod
    def discretize():
        '''Takes DomainDescription and returns tuple (AffineGrid, BoundaryInfo)'''
        pass

