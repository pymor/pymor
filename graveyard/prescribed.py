# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import division
import math
import numpy as np
from matplotlib.delaunay import delaunay
from pymor.core import cache, Unpicklable
from pymor.grids.interfaces import AffineGridInterface
from pymor.grids import referenceelements as Refs
from pymor.core.exceptions import CodimError
from pymor.grids.defaultimpl import ConformalTopologicalGridDefaultImplementations


class CircleBoundary(object):

    def __init__(self, radius=1, center=(0, 0)):
        self.radius = radius
        self.center = center

    def __call__(self, p):
        '''for p in 0,1 returns the point on S1 in R2'''
        phi = 2 * math.pi * p
        return (self.center[0] + self.radius * math.sin(phi), self.center[1] + self.radius * math.cos(phi))


class StructuredSimplexCheat(object):
    def __init__(self, points, h):
        self.count = 0
        self.points = points
        self.rows = math.sqrt(points)
        self.cols = self.rows
        self.h = h

    def __call__(self, p):
        x = (self.count % self.cols) * self.h
        y = (self.count // self.rows) * self.h
        self.count += 1
        if self.count >= self.points:
            self.count = 0
        return (x, y)


class PrescribedBoundaryGrid(AffineGridInterface, cache.Cachable, Unpicklable):
    '''given an analytical boundary descriptjon f: [0,1)mapto R2
    I'll sample n points and construct a simplicial mesh of their convex hull
    '''
    dim = 2
    dim_outer = 2
    _ref_elements = {0: Refs.triangle, 1: Refs.line, 2: Refs.point}

    class Edge(object):
        def __init__(self, idA, idB):
            if idA > idB:
                self._idA = idA
                self._idB = idB
            else:
                self._idA = idB
                self._idB = idA

        def __eq__(self, other):
            return ((self._idA == other._idA) and (self._idB == other._idB)
                    or ((self._idB == other._idA) and (self._idA == other._idB)))

        def __hash__(self, *args, **kwargs):
            return self._idA * 100 + self._idB

        def __repr__(self):
            return 'Edge {} - {}'.format(self._idA, self._idB)

    class Triangle(object):
        def __init__(self, ids, edges):
            Ed = PrescribedBoundaryGrid.Edge
            self.edges = sorted([edges.index(Ed(ids[i[0]], ids[i[1]])) for i in [(0, 1), (1, 2), (2, 0)]])
            self.vertices = sorted(ids)

    def __init__(self, boundary_callable=CircleBoundary(), sample_count=20):
        '''
        :param boundary_callable: a callable we can sample sample_count times in [0,1)
        :param sample_count: how many points to sample from boundary_callable
        if boundary function is of that fancy cas package we could sample
        weighed by the absolute value of the gradient. otherwise sampling is uniform
        '''
        super(PrescribedBoundaryGrid, self).__init__()
        fac = 1 / sample_count
        self._px = np.array([boundary_callable(p * fac)[0] for p in range(sample_count)])
        self._py = np.array([boundary_callable(p * fac)[1] for p in range(sample_count)])
        self._vertices = zip(self._px, self._py)

        self._centers, _, triangles, self._neighbours_ids = delaunay(self._px, self._py)
        self._edges = []
        for ids in triangles:
            self._edges.extend([self.Edge(ids[i[0]], ids[i[1]]) for i in [(0, 1), (1, 2), (2, 0)]])

        self._triangles = [self.Triangle(x, self._edges) for x in triangles]
        self._sizes = {0: len(self._triangles), 1: len(self._edges), 2: len(self._px)}
        self.__subentities = list(xrange(self.dim + 1))
        self.__subentities[2] = np.array([c.vertices for c in self._triangles], dtype='int32')
        self.__subentities[1] = np.array([c.edges for c in self._triangles], dtype='int32')
        self.__subentities[0] = np.arange(len(self._triangles), dtype='int32')[:, np.newaxis]
        self.__embed_A = np.empty((self._sizes[0], 2, 2))
        self.__embed_B = np.empty((self._sizes[0], 2))

        for i, triangle in enumerate(self._triangles):
            x = [(self._px[triangle.vertices[j]], self._py[triangle.vertices[j]]) for j in (0, 1, 2)]
            W = np.array([[x[0][0], x[0][1], 0,       0,       1, 0],
                          [0,       0,       x[0][0], x[0][1], 0, 1],
                          [x[1][0], x[1][1], 0,       0,       1, 0],
                          [0,       0,       x[1][0], x[1][1], 0, 1],
                          [x[2][0], x[2][1], 0,       0,       1, 0],
                          [0,       0,       x[2][0], x[2][1], 0, 1]])
            K = np.array([0, 0, 0, 1, 1, 1])
            try:
                R = np.linalg.solve(W, K)
            except np.linalg.linalg.LinAlgError, e:
                import pprint
                self.logger.critical(pprint.pformat(K))
                self.logger.critical(pprint.pformat(W))
                self.logger.error(pprint.pformat(self._vertices))
                raise e
            self.__embed_A[i] = [[R[0], R[1]],
                                 [R[2], R[3]]]
            self.__embed_B[i] = [R[4], R[5]]
        for i, k in self.__dict__.iteritems():
            try:
                self.logger.debug('{} len {}'.format(i, len(k)))
            except:
                pass
        for i, k in enumerate(self.__subentities):
            self.logger.debug('Subentities {} len {}'.format(i, len(k)))
        assert(self.Edge(1, 0) == self.Edge(0, 1))

    def subentities(self, codim, subentity_codim=None):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension')
        assert codim <= subentity_codim <= self.dim, CodimError('Invalid subentity codimensoin')
        subentity_codim = subentity_codim or codim + 1
        if codim == 0:
            return self.__subentities[subentity_codim]
        else:
            return self._subentities(codim, subentity_codim)

    def reference_element(self, codim):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension')
        return self._ref_elements[codim]

    def size(self, codim=0):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension')
        return self._sizes[codim]

    def embeddings(self, codim):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension')
        if codim > 0:
            return super(PrescribedBoundaryGrid, self)._embeddings(codim)
        return (self.__embed_A, self.__embed_B)

    def plot(self):
        import pylab
        for t in self._triangles:
            t = t.vertices  # [t[i] for i in range(-1, 3)]
#            pylab.plot(self._px[t], self._py[t])
#            pylab.plot(self._px[t], self._py[t],'o')
        for _, t in enumerate(self._edges):
            t = [t._idA, t._idB]
            pylab.plot(self._px[t], self._py[t])
#            pylab.plot(self._px[t], self._py[t],'x')
        pylab.show()
#        for t in self.subentities(1):
#            t = [t[i] for i in range(2)]
#            pylab.plot(self._px[t], self._py[t])
#        pylab.show()
#        for t in self.subentities(2):
#            pylab.plot(self._px[t], self._py[t],'o')
#        pylab.show()

    @staticmethod
    def test_instances():
        return [PrescribedBoundaryGrid()]

if __name__ == "__main__":
    gr = PrescribedBoundaryGrid(CircleBoundary(), sample_count=12)
    gr.plot()
    c = 25
#    gr = PrescribedBoundaryGrid(StructuredSimplexCheat(c, 0.1), sample_count=c)
#    gr.plot()
