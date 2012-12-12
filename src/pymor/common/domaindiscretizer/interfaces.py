from __future__ import absolute_import, division, print_function, unicode_literals

import pymor.core as core


class IDomainDiscretizer(core.BasicInterface):
    '''Takes IDomainDiscription and generates IGrid and IBoundaryInfo'''

    @core.interfaces.abstractmethod
    def discretize(domain_description):
        '''Takes IDomainDescription and returns tuple (IGrid, IBoundaryInfo)'''
        pass

