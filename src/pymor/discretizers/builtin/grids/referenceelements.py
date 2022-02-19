# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.discretizers.builtin.grids.interfaces import ReferenceElement
from pymor.discretizers.builtin.quadratures import GaussQuadratures


class Point(ReferenceElement):

    dim = 0
    volume = 1

    def size(self, codim):
        assert codim == 0, f'Invalid codimension (must be 0 but was {codim})'
        return 1

    def subentities(self, codim, subentity_codim):
        assert codim == 0, f'Invalid codimension (must be 0 but was {codim})'
        assert subentity_codim == 0, f'Invalid subentity codimension (must be 0 but was {subentity_codim})'
        return np.array([0], dtype='int32')

    def subentity_embedding(self, subentity_codim):
        assert subentity_codim == 0, f'Invalid codimension (must be 0 but was {subentity_codim})'
        return np.zeros((0, 0), dtype='int32'), np.zeros((0), dtype='int32')

    def sub_reference_element(self, codim):
        assert codim == 0, f'Invalid codimension (must be 0 but was {codim})'
        return self

    def unit_outer_normals(self):
        return np.zeros((0))

    def center(self):
        return np.zeros((0))

    def mapped_diameter(self, A):
        return np.ones(A.shape[:-2])

    def quadrature_info(self):
        # of course, the quadrature is of arbitrary oder ...
        return {'gauss': tuple(range(42))}, {'gauss': (1,)}

    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        if quadrature_type == 'default' or quadrature_type == 'gauss':
            assert npoints is None or npoints == 1, 'there is only one point in dimension 0!'
            return np.zeros((1, 0)), np.ones(1)
        else:
            raise NotImplementedError('quadrature_type must be "default" or "gauss"')


point = Point()


class Line(ReferenceElement):

    dim = 1
    volume = 1

    def size(self, codim):
        assert 0 <= codim <= 1, f'Invalid codimension (must be 0 or 1 but was {codim})'
        if codim == 0:
            return 1
        else:
            return 2

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 1, f'Invalid codimension (must be 0 or 1 but was {codim})'
        assert codim <= subentity_codim <= 1,\
            f'Invalid codimension (must be between {codim} and 1 but was {subentity_codim})'
        if codim == 0:
            return np.arange(self.size(subentity_codim), dtype='int32')
        else:
            return np.array(([0], [1]), dtype='int32')

    def subentity_embedding(self, subentity_codim):
        assert 0 <= subentity_codim <= 1,\
            f'Invalid codimension (must be 0 or 1 but was {subentity_codim})'
        if subentity_codim == 0:
            return np.ones((1, 1, 1)), np.zeros((1, 1, 1))
        else:
            return np.array((np.zeros((1, 0)), np.zeros((1, 0)))), np.array(([0.], [1.]))

    def sub_reference_element(self, codim):
        assert 0 <= codim <= 1, f'Invalid codimension (must be 0 or 1 but was {codim})'
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
        return {'gauss': GaussQuadratures.orders}, {'gauss': list(map(len, GaussQuadratures.points))}

    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        if quadrature_type == 'default' or quadrature_type == 'gauss':
            P, W = GaussQuadratures.quadrature(order, npoints)
            return P[:, np.newaxis], W
        else:
            raise NotImplementedError('quadrature_type must be "default" or "gauss"')


line = Line()


class Square(ReferenceElement):

    dim = 2
    volume = 1

    def __init__(self):

        def tensor_points(P):
            PP0, PP1 = np.array(np.meshgrid(P, P))
            return np.array((PP0.ravel(), PP1.ravel())).T

        def tensor_weights(W):
            return np.dot(W[:, np.newaxis], W[np.newaxis, :]).ravel()
        self._quadrature_points = [tensor_points(GaussQuadratures.quadrature(npoints=p + 1)[0])
                                   for p in range(GaussQuadratures.maxpoints())]
        self._quadrature_weights = [tensor_weights(GaussQuadratures.quadrature(npoints=p + 1)[1])
                                    for p in range(GaussQuadratures.maxpoints())]
        self._quadrature_npoints = np.arange(1, GaussQuadratures.maxpoints() + 1) ** 2
        self._quadrature_orders = GaussQuadratures.orders
        self._quadrature_order_map = GaussQuadratures.order_map

    def size(self, codim):
        assert 0 <= codim <= 2, f'Invalid codimension (must be between 0 and 2 but was {codim})'
        if codim == 0:
            return 1
        elif codim == 1:
            return 4
        elif codim == 2:
            return 4

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 2, f'Invalid codimension (must be between 0 and 2 but was {codim})'
        assert codim <= subentity_codim <= 2,\
            f'Invalid codimension (must be between {codim} and 2 but was {subentity_codim})'
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
            f'Invalid codimension (must between 0 and 2 but was {subentity_codim})'
        if subentity_codim == 0:
            return np.eye(2), np.zeros(2)
        elif subentity_codim == 1:
            A = np.array((np.array(([1.], [0.])), np.array(([0.], [1.])),
                          np.array(([-1.], [0.])), np.array(([0.], [-1.]))))
            B = np.array((np.array([0., 0.]), np.array([1., 0.]),
                          np.array([1., 1.]), np.array([0., 1.])))
            return A, B
        else:
            return super().subentity_embedding(subentity_codim)

    def sub_reference_element(self, codim):
        assert 0 <= codim <= 2, f'Invalid codimension (must be between 0 and 2 but was {codim})'
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
            assert order is not None or npoints is not None, 'must specify "order" or "npoints"'
            assert order is None or npoints is None, 'cannot specify "order" and "npoints"'
            if order is not None:
                assert 0 <= order <= self._quadrature_order_map.size - 1,\
                    ValueError(f'order {order} not implmented')
                p = self._quadrature_order_map[order]
            else:
                assert npoints in self._quadrature_npoints, f'not implemented with {npoints} points'
                p = np.where(self._quadrature_npoints == npoints)[0][0]
            return self._quadrature_points[p], self._quadrature_weights[p]
        else:
            raise NotImplementedError('quadrature_type must be "default" or "tensored_gauss"')


square = Square()


class Triangle(ReferenceElement):

    dim = 2
    volume = 0.5

    def size(self, codim):
        assert 0 <= codim <= 2, f'Invalid codimension (must be between 0 and 2 but was {codim})'
        if codim == 0:
            return 1
        elif codim == 1:
            return 3
        elif codim == 2:
            return 3

    def subentities(self, codim, subentity_codim):
        assert 0 <= codim <= 2, f'Invalid codimension (must be between 0 and 2 but was {codim})'
        assert codim <= subentity_codim <= 2,\
            f'Invalid codimension (must be between {codim} and 2 but was {subentity_codim})'
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
            f'Invalid codimension (must between 0 and 2 but was {subentity_codim})'
        if subentity_codim == 0:
            return np.eye(2), np.zeros(2)
        elif subentity_codim == 1:
            A = np.array((np.array(([-1.], [1.])), np.array(([0.], [-1.])),
                          np.array(([1.], [0.]))))
            B = np.array((np.array([1., 0.]), np.array([0., 1.]),
                          np.array([0., 0.])))
            return A, B
        else:
            return super().subentity_embedding(subentity_codim)

    def sub_reference_element(self, codim):
        assert 0 <= codim <= 2, f'Invalid codimension (must be between 0 and 2 but was {codim})'
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
        assert order is not None or npoints is not None, 'must specify "order" or "npoints"'
        assert order is None or npoints is None, 'cannot specify "order" and "npoints"'
        if quadrature_type == 'default':
            if order == 1 or npoints == 1:
                quadrature_type = 'center'
            else:
                quadrature_type = 'edge_centers'

        if quadrature_type == 'center':
            assert order is None or order == 1
            assert npoints is None or npoints == 1
            return np.array([self.center()]), np.array([self.volume])
        elif quadrature_type == 'edge_centers':
            assert order is None or order <= 2
            assert npoints is None or npoints == 3
            # this would work for arbitrary reference elements
            # L, A = self.subentity_embedding(1)
            # return (np.array(L.dot(self.sub_reference_element().center()) + A),
            #         np.ones(3) / len(A) * self.volume)
            return np.array(([0.5, 0.5], [0, 0.5], [0.5, 0])), np.ones(3) / 3 * self.volume
        else:
            raise NotImplementedError('quadrature_type must be "center" or "edge_centers"')


triangle = Triangle()
