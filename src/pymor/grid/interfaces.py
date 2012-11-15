from __future__ import absolute_import, division, print_function, unicode_literals

import abc

import pymor.core as core


class IGrid(core.BaseInterface):
    '''Base interface for all grids. This is an incomplete prepreliminary version.
    Until now, only the toplogy part of the interface is specified in here.
    '''

    @abc.abstractmethod
    def size(self, codim=0):
        '''size(codim) is the number of entities in the grid of codimension codim'''
        pass

    @abc.abstractmethod
    def subentities(self, codim=0, subentity_codim=None):
        '''retval[s,e] is the global index of the s-th codim-"subentity_codim"
        subentity of the codim-"codim" entity with global index e.

        If subentity_codim=None, it is set to codim+1.
        '''
        pass

    @abc.abstractmethod
    def superentities(self, codim, superentity_codim=None):
        '''retval[s,e] is the global index of the s-th codim-"superentity_codim"
        superentity of the codim-"codim" entity with global index e.

        If superentity_codim == None, it is set to codim-1.
        '''
        pass

    @abc.abstractmethod
    def neighbours(self, codim=0, neighbour_codim=0, intersection_codim=None):
        '''retval[n,e] is the global index of the n-th codim-"neighbour_codim"
        entitiy of the codim-"codim" entity with global index e that shares
        with it an intersection of codimension "intersection_codim".

        If intersection_codim == None,
            it is set to codim if codim == neighbour_codim
            otherwise it is set to min(codim, neighbour_codim).
        '''
        pass
