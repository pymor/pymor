from __future__ import absolute_import, division, print_function, unicode_literals

import pymor.core as core


class DomainDiscretizerInterface(core.BasicInterface):
    '''Takes DomainDiscriptionInterface and generates GridInterface and BoundaryInfoInterface'''

    @core.interfaces.abstractmethod
    def discretize(domain_description):
        '''Takes DomainDescriptionInterface and returns tuple (GridInterface, BoundaryInfoInterface)'''
        pass

