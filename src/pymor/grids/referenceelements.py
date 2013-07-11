# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from pymor.core.exceptions import CodimError
from pymor.tools.quadratures import GaussQuadratures
from .interfaces import ReferenceElementInterface


class Point(ReferenceElementInterface):

    dim = 0
    volume = 1

    def __init__(self):
        super(Point, self).__init__()
        self.lock()

    def size(self, codim=1):
        assert codim == 0, CodimError('Invalid codimension (must be 0 but was {})'.format(codim))
        return 1

    def subentities(self, codim, subentity_codim):
        assert codim == 0, CodimError('Invalid codimension (must be 0 but was {})'.format(codim))
        assert subentity_codim == 0, CodimError('Invalid subentity codimension (must be 0 but was {})'
                                                .format(subentity_codim))
        return np.array([0], dtype='int32')

    def subentity_embedding(self, subentity_codim):
        assert subentity_codim == 0, CodimError('Invalid codimension (must be 0 but was {})'.format(subentity_codim))
        return np.zeros((0, 0), dtype='int32'), np.zeros((0), dtype='int32')

    def sub_reference_element(self, codim=1):
        assert codim == 0, CodimError('Invalid codimension (must be 0 but was {})'.format(codim))
        return self

    def unit_outer_normals(self):
        return np.zeros((0))

    def center(self):
        return np.zeros((0))

    def mapped_diameter(self, A):
        return np.ones(A.shape[:-2])

    def quadrature_info(self):
        # of course, the quadrature is of abritrary oder ...
        return {'gauss': tuple(xrange(42))}, {'gauss': (1,)}

    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        if quadrature_type == 'default' or quadrature_type == 'gauss':
            assert npoints is None or npoints == 1, ValueError('there is only one point in dimension 0!')
            return np.zeros((1, 0)), np.ones(1)
        else:
            raise NotImplementedError('quadrature_type must be "default" or "gauss"')

point = Point()


class Line(ReferenceElementInterface):

    dim = 1
    volume = 1

    def __init__(self):
        super(Line, self).__init__()
        self.lock()

    def size(self, codim=1):
        assert 0 <= codim <= 1, CodimError('Invalid codimension (must be 0 or 1 but was {})'.format(codim))
        if codim == 0:
            return 1
        else:
            return 2

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 1, CodimError('Invalid codimension (must be 0 or 1 but was {})'.format(codim))
        assert codim <= subentity_codim <= 1,\
            CodimError('Invalid codimension (must be between {} and 1 but was {})'.format(codim, subentity_codim))
        if codim == 0:
            return np.arange(self.size(subentity_codim), dtype='int32')
        else:
            return np.array(([0], [1]), dtype='int32')

    def subentity_embedding(self, subentity_codim):
        assert 0 <= subentity_codim <= 1,\
            CodimError('Invalid codimension (must be 0 or 1 but was {})'.format(subentity_codim))
        if subentity_codim == 0:
            return np.ones((1, 1, 1)), np.zeros((1, 1, 1))
        else:
            return np.array((np.zeros((1, 0)), np.zeros((1, 0)))), np.array(([0.], [1.]))

    def sub_reference_element(self, codim=1):
        assert 0 <= codim <= 1, CodimError('Invalid codimension (must be 0 or 1 but was {})'.format(codim))
        if codim == 0:
            return self
        else:
            return point

    def unit_outer_normals(self):
        return np.array(([-1.], [1.]))

    def center(self):
        return np.array([0.5])

    def mapped_diameter(self, A):
        return np.apply_along_axis(np.linalg.norm, -2, A)

    def quadrature_info(self):
        return {'gauss': GaussQuadratures.orders}, {'gauss': map(len, GaussQuadratures.points)}

    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        if quadrature_type == 'default' or quadrature_type == 'gauss':
            P, W = GaussQuadratures.quadrature(order, npoints)
            return P[:, np.newaxis], W
        else:
            raise NotImplementedError('quadrature_type must be "default" or "gauss"')

line = Line()


class Square(ReferenceElementInterface):

    dim = 2
    volume = 1

    def __init__(self):
        super(Square, self).__init__()
        def tensor_points(P):
            PP0, PP1 = np.array(np.meshgrid(P, P))
            return np.array((PP0.ravel(), PP1.ravel())).T
        self.lock()

        def tensor_weights(W):
            return np.dot(W[:, np.newaxis], W[np.newaxis, :]).ravel()
        self._quadrature_points = [tensor_points(GaussQuadratures.quadrature(npoints=p + 1)[0])
                                   for p in xrange(GaussQuadratures.maxpoints())]
        self._quadrature_weights = [tensor_weights(GaussQuadratures.quadrature(npoints=p + 1)[1])
                                    for p in xrange(GaussQuadratures.maxpoints())]
        self._quadrature_npoints = np.arange(1, GaussQuadratures.maxpoints() + 1) ** 2
        self._quadrature_orders = GaussQuadratures.orders
        self._quadrature_order_map = GaussQuadratures.order_map

    def size(self, codim=1):
        assert 0 <= codim <= 2, CodimError('Invalid codimension (must be between 0 and 2 but was {})'.format(codim))
        if codim == 0:
            return 1
        elif codim == 1:
            return 4
        elif codim == 2:
            return 4

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 2, CodimError('Invalid codimension (must be between 0 and 2 but was {})'.format(codim))
        assert codim <= subentity_codim <= 2,\
            CodimError('Invalid codimension (must be between {} and 2 but was {})'.format(codim, subentity_codim))
        if codim == 0:
            return np.arange(self.size(subentity_codim), dtype='int32')
        elif codim == 1:
            if subentity_codim == 1:
                return np.array(([0], [1], [2], [3]), dtype='int32')
            else:
                return np.array(([0, 1], [1, 2], [2, 3], [3, 0]), dtype='int32')
        elif codim == 2:
            return np.array(([0], [1], [2], [3]), dtype='int32')

    def subentity_embedding(self, subentity_codim):
        assert 0 <= subentity_codim <= 2,\
            CodimError('Invalid codimension (must betwen 0 and 2 but was {})'.format(subentity_codim))
        if subentity_codim == 0:
            return np.eye(2), np.zeros(2)
        elif subentity_codim == 1:
            A = np.array((np.array(([1.], [0.])), np.array(([0.], [1.])),
                          np.array(([-1.], [0.])), np.array(([0.], [-1.]))))
            B = np.array((np.array([0., 0.]), np.array([1., 0.]),
                          np.array([1., 1.]), np.array([0., 1.])))
            return A, B
        else:
            return super(Square, self).subentity_embedding(subentity_codim)

    def sub_reference_element(self, codim=1):
        assert 0 <= codim <= 2, CodimError('Invalid codimension (must be between 0 and 2 but was {})'.format(codim))
        if codim == 0:
            return self
        elif codim == 1:
            return line
        else:
            return point

    def unit_outer_normals(self):
        return np.array(([0., -1.], [1., 0.], [0., 1], [-1., 0.]))

    def center(self):
        return np.array([0.5, 0.5])

    def mapped_diameter(self, A):
        V0 = np.dot(A, np.array([1., 1.]))
        V1 = np.dot(A, np.array([1., -1]))
        VN0 = np.apply_along_axis(np.linalg.norm, -1, V0)
        VN1 = np.apply_along_axis(np.linalg.norm, -1, V1)
        return np.max((VN0, VN1), axis=0)

    def quadrature_info(self):
        return {'tensored_gauss': self._quadrature_orders}, {'tensored_gauss': self._quadrature_npoints}

    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        if quadrature_type == 'default' or quadrature_type == 'tensored_gauss':
            assert order is not None or npoints is not None, ValueError('must specify "order" or "npoints"')
            assert order is None or npoints is None, ValueError('cannot specify "order" and "npoints"')
            if order is not None:
                assert 0 <= order <= self._quadrature_order_map.size - 1,\
                    ValueError('order {} not implmented'.format(order))
                p = self._quadrature_order_map[order]
            else:
                assert npoints in self._quadrature_npoints, ValueError('not implemented with {} points'.format(npoints))
                p = np.where(self._quadrature_npoints == npoints)[0]
            return self._quadrature_points[p], self._quadrature_weights[p]
        else:
            raise NotImplementedError('quadrature_type must be "default" or "tensored_gauss"')


square = Square()


class Triangle(ReferenceElementInterface):

    dim = 2
    volume = 0.5

    def __init__(self):
        super(Triangle, self).__init__()
        self.lock()
        # def tensor_points(P):
            # PP0, PP1 = np.array(np.meshgrid(P, P))
            # return np.array((PP0.ravel(), PP1.ravel())).T

        # def tensor_weights(W):
            # return np.dot(W[:,np.newaxis], W[np.newaxis, :]).ravel()
        # self._quadrature_points  = [tensor_points(Gauss.quadrature(npoints=p+1)[0])
                                      # for p in xrange(Gauss.maxpoints())]
        # self._quadrature_weights = [tensor_weights(Gauss.quadrature(npoints=p+1)[1])
                                      # for p in xrange(Gauss.maxpoints())]
        # self._quadrature_npoints = np.arange(1, Gauss.maxpoints() + 1) ** 2
        # self._quadrature_orders  = Gauss.orders
        # self._quadrature_order_map = Gauss.order_map

    def size(self, codim=1):
        assert 0 <= codim <= 2, CodimError('Invalid codimension (must be between 0 and 2 but was {})'.format(codim))
        if codim == 0:
            return 1
        elif codim == 1:
            return 3
        elif codim == 2:
            return 3

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 2, CodimError('Invalid codimension (must be between 0 and 2 but was {})'.format(codim))
        assert codim <= subentity_codim <= 2,\
            CodimError('Invalid codimension (must be between {} and 2 but was {})'.format(codim, subentity_codim))
        if codim == 0:
            return np.arange(self.size(subentity_codim), dtype='int32')
        elif codim == 1:
            if subentity_codim == 1:
                return np.array(([0], [1], [2]), dtype='int32')
            else:
                return np.array(([1, 2], [2, 0], [0, 1]), dtype='int32')
        elif codim == 2:
            return np.array(([0], [1], [2]), dtype='int32')

    def subentity_embedding(self, subentity_codim):
        assert 0 <= subentity_codim <= 2,\
            CodimError('Invalid codimension (must betwen 0 and 2 but was {})'.format(subentity_codim))
        if subentity_codim == 0:
            return np.eye(2), np.zeros(2)
        elif subentity_codim == 1:
            A = np.array((np.array(([-1.], [1.])), np.array(([0.], [-1.])),
                          np.array(([1.], [0.]))))
            B = np.array((np.array([1., 0.]), np.array([0., 1.]),
                          np.array([0., 0.])))
            return A, B
        else:
            return super(Triangle, self).subentity_embedding(subentity_codim)

    def sub_reference_element(self, codim=1):
        assert 0 <= codim <= 2, CodimError('Invalid codimension (must be between 0 and 2 but was {})'.format(codim))
        if codim == 0:
            return self
        elif codim == 1:
            return line
        else:
            return point

    def unit_outer_normals(self):
        return np.array(([1., 1.], [-1., 0.], [0., -1.]))

    def center(self):
        return np.array([1. / 3., 1. / 3.])

    def mapped_diameter(self, A):
        V0 = np.dot(A, np.array([-1., 1.]))
        V1 = np.dot(A, np.array([0., -1.]))
        V2 = np.dot(A, np.array([1., 0.]))
        VN0 = np.apply_along_axis(np.linalg.norm, -1, V0)
        VN1 = np.apply_along_axis(np.linalg.norm, -1, V1)
        VN2 = np.apply_along_axis(np.linalg.norm, -1, V2)
        return np.max((VN0, VN1, VN2), axis=0)

    def quadrature_info(self):
        return ({'center': (1,), 'edge_centers': (2,)},
                {'center': (1,), 'edge_centers': (3,)})

    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        assert order is not None or npoints is not None, ValueError('must specify "order" or "npoints"')
        assert order is None or npoints is None, ValueError('cannot specify "order" and "npoints"')
        if quadrature_type == 'default':
            if order == 1 or npoints == 1:
                quadrature_type = 'center'
            else:
                quadrature_type = 'edge_centers'

        if quadrature_type == 'center':
            assert order is None or order == 1
            assert npoints is None or npoints == 1
            return np.array((self.center(),)), np.array(self.volume)
        elif quadrature_type == 'edge_centers':
            assert order is None or order <= 2
            assert npoints is None or npoints == 3
            #this would work for arbitrary reference elements
            #L, A = self.subentity_embedding(1)
            #return np.array(L.dot(self.sub_reference_element().center()) + A), np.ones(3) / len(A) * self.volume
            return np.array(([0.5, 0.5], [0, 0.5], [0.5, 0])), np.ones(3) / 3 * self.volume
        else:
            raise NotImplementedError('quadrature_type must be "center" or "edge_centers"')


triangle = Triangle()
