# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, product
import numpy as np

from pymor.grids.interfaces import (ConformalTopologicalGridInterface, AffineGridInterface, ReferenceElementInterface)
# mandatory so all Grid classes are created
from pymor.grids import *

from pymortests.base import (runmodule, GridClassTestInterface, GridSubclassForImplemetorsOf)


@GridSubclassForImplemetorsOf(ConformalTopologicalGridInterface)
class ConformalTopologicalGridTestInterface(GridClassTestInterface):

    def test_dim(self):
        for g in self.grids:
            self.assertIsInstance(g.dim, int)
            self.assertGreaterEqual(g.dim, 0)

    def test_size(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertIsInstance(g.size(d), int)
                self.assertGreaterEqual(g.size(d), 0)
            with self.assertRaises(AssertionError):
                g.size(-1)
            with self.assertRaises(AssertionError):
                g.size(g.dim + 1)

    def test_subentities_wrong_arguments(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                with self.assertRaises(AssertionError):
                    g.subentities(e, g.dim + 1)
                with self.assertRaises(AssertionError):
                    g.subentities(e, e - 1)
            with self.assertRaises(AssertionError):
                g.subentities(g.dim + 1, 0)
            with self.assertRaises(AssertionError):
                g.subentities(-1, 0)

    def test_subentities_shape(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
                    self.assertEqual(g.subentities(e, s).ndim, 2)
                    self.assertEqual(g.subentities(e, s).shape[0], g.size(e))

    def test_subentities_dtype(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
                    self.assertEqual(g.subentities(e, s).dtype, np.dtype('int32'),
                                     'Failed for\n{g}\ne={e}, s={s}'.format(**locals()))

    def test_subentities_entry_value_range(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
                    self.logger.debug('E {} -- S {}'.format(e, s))
                    np.testing.assert_array_less(g.subentities(e, s), g.size(s))
                    np.testing.assert_array_less(-2, g.subentities(e, s))

    def test_subentities_entry_values_unique(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
                    for S in g.subentities(e, s):
                        S = S[S >= 0]
                        self.assertEqual(S.size, np.unique(S).size)

    def test_subentities_codim_d_codim_d(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertEqual(g.subentities(d, d).shape, (g.size(d), 1))
                np.testing.assert_array_equal(g.subentities(d, d).ravel(), np.arange(g.size(d)))

    def test_subentities_transitivity(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e + 1, g.dim + 1):
                    for ss in xrange(s + 1, g.dim + 1):
                        SE = g.subentities(e, s)
                        SSE = g.subentities(e, ss)
                        SESE = g.subentities(s, ss)
                        for i in xrange(SE.shape[0]):
                            for j in xrange(SE.shape[1]):
                                if SE[i, j] != -1:
                                    self.assertTrue(set(SESE[SE[i, j]]) <= set(SSE[i]).union((-1,)))

    def test_superentities_wrong_arguments(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                with self.assertRaises(AssertionError):
                    g.superentities(e, -1)
                with self.assertRaises(AssertionError):
                    g.superentities(e, e + 1)
            with self.assertRaises(AssertionError):
                g.superentities(g.dim + 1, 0)
            with self.assertRaises(AssertionError):
                g.superentities(-1, 0)
            with self.assertRaises(AssertionError):
                g.superentities(-1, -2)

    def test_superentities_shape(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    self.assertEqual(g.superentities(e, s).ndim, 2)
                    self.assertEqual(g.superentities(e, s).shape[0], g.size(e))
                    self.assertGreater(g.superentities(e, s).shape[1], 0)
            self.assertLessEqual(g.superentities(1, 0).shape[1], 2)

    def test_superentities_dtype(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    self.assertEqual(g.superentities(e, s).dtype, np.dtype('int32'))

    def test_superentities_entry_value_range(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    np.testing.assert_array_less(g.superentities(e, s), g.size(s))
                    np.testing.assert_array_less(-2, g.superentities(e, s))

    def test_superentities_entry_values_unique(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    for S in g.superentities(e, s):
                        S = S[S >= 0]
                        self.assertEqual(S.size, np.unique(S).size)

    def test_superentities_entries_sorted(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    for S in g.superentities(e, s):
                        i = 0
                        while (i + 1 < len(S)) and (S[i] < S[i + 1]):
                            i += 1
                        self.assertTrue((i + 1 == len(S)) or (S[i + 1] == -1))
                        if i + 1 < len(S):
                            np.testing.assert_array_equal(S[i + 1:], -1)

    def test_superentities_codim_d_codim_d(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertEqual(g.superentities(d, d).shape, (g.size(d), 1))
                np.testing.assert_array_equal(g.superentities(d, d).ravel(), np.arange(g.size(d)))

    def test_superentities_each_entry_superentity(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    SE = g.superentities(e, s)
                    SUBE = g.subentities(s, e)
                    for i in xrange(SE.shape[0]):
                        for se in SE[i]:
                            if se != -1:
                                self.assertTrue(i in SUBE[se])

    def test_superentities_each_superentity_has_entry(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    SE = g.superentities(e, s)
                    SUBE = g.subentities(s, e)
                    for i in xrange(SUBE.shape[0]):
                        for se in SUBE[i]:
                            if se != -1:
                                self.assertTrue(i in SE[se])

    def test_superentity_indices_wrong_arguments(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                with self.assertRaises(AssertionError):
                    g.superentity_indices(e, -1)
                with self.assertRaises(AssertionError):
                    g.superentity_indices(e, e + 1)
            with self.assertRaises(AssertionError):
                g.superentity_indices(g.dim + 1, 0)
            with self.assertRaises(AssertionError):
                g.superentity_indices(-1, 0)
            with self.assertRaises(AssertionError):
                g.superentity_indices(-1, -2)

    def test_superentity_indices_shape(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    self.assertEqual(g.superentity_indices(e, s).shape, g.superentities(e, s).shape)

    def test_superentity_indices_dtype(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    self.assertEqual(g.superentity_indices(e, s).dtype, np.dtype('int32'))

    def test_superentity_indices_valid_entries(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e):
                    SE = g.superentities(e, s)
                    SEI = g.superentity_indices(e, s)
                    SUBE = g.subentities(s, e)
                    for index, superentity in np.ndenumerate(SE):
                        if superentity > -1:
                            self.assertEqual(SUBE[superentity, SEI[index]], index[0])

    def test_neighbours_wrong_arguments(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for n in xrange(g.dim + 1):
                    with self.assertRaises(AssertionError):
                        g.neighbours(e, n, -1)
                    with self.assertRaises(AssertionError):
                        g.neighbours(e, n, g.dim + 1)
                    with self.assertRaises(AssertionError):
                        g.neighbours(e, n, e - 1)
                    with self.assertRaises(AssertionError):
                        g.neighbours(e, n, n - 1)
                with self.assertRaises(AssertionError):
                    g.neighbours(e, g.dim + 1, g.dim)
                with self.assertRaises(AssertionError):
                    g.neighbours(e, -1, g.dim)
            with self.assertRaises(AssertionError):
                g.neighbours(g.dim + 1, g.dim, g.dim)
            with self.assertRaises(AssertionError):
                g.neighbours(-1, 0, g.dim)

    def test_neighbours_shape(self):
        for g in self.grids:
            for e, n in product(xrange(g.dim + 1), xrange(g.dim + 1)):
                for s in xrange(max(e, n), g.dim + 1):
                    self.assertEqual(g.neighbours(e, n, s).ndim, 2)
                    self.assertEqual(g.neighbours(e, n, s).shape[0], g.size(e))

    def test_neighbours_dtype(self):
        for g in self.grids:
            for e, n in product(xrange(g.dim + 1), xrange(g.dim + 1)):
                for s in xrange(max(e, n), g.dim + 1):
                    self.assertEqual(g.neighbours(e, n, s).dtype, np.dtype('int32'))

    def test_neighbours_entry_value_range(self):
        for g in self.grids:
            for e, n in product(xrange(g.dim + 1), xrange(g.dim + 1)):
                for s in xrange(max(e, n), g.dim + 1):
                    np.testing.assert_array_less(g.neighbours(e, n, s), g.size(n))
                    np.testing.assert_array_less(-2, g.neighbours(e, n, s))

    def test_neighbours_entry_values_unique(self):
        for g in self.grids:
            for e, n in product(xrange(g.dim + 1), xrange(g.dim + 1)):
                for s in xrange(max(e, n), g.dim + 1):
                    for S in g.neighbours(e, n, s):
                        S = S[S >= 0]
                        self.assertEqual(S.size, np.unique(S).size)

    def test_neighbours_each_entry_neighbour(self):
        for g in self.grids:
            for e, n in product(xrange(g.dim + 1), xrange(g.dim + 1)):
                for s in xrange(max(e, n), g.dim + 1):
                    N = g.neighbours(e, n, s)
                    ESE = g.subentities(e, s)
                    NSE = g.subentities(n, s)
                    for index, neigh in np.ndenumerate(N):
                        if neigh > -1:
                            inter = set(ESE[index[0]]).intersection(set(NSE[neigh]))
                            if -1 in inter:
                                self.assertTrue(len(inter) > 1)
                            else:
                                self.assertTrue(len(inter) > 0)

    def test_neighbours_each_neighbour_has_entry(self):
        for g in self.grids:
            for e, n in product(xrange(g.dim + 1), xrange(g.dim + 1)):
                for s in xrange(max(e, n), g.dim + 1):
                    N = g.neighbours(e, n, s)
                    SUE = g.superentities(s, e)
                    SUN = g.superentities(s, n)
                    if e != n:
                        for si in xrange(SUE.shape[0]):
                            for ei, ni in product(SUE[si], SUN[si]):
                                if ei != -1 and ni != -1:
                                    self.assertTrue(ni in N[ei],
                                                    'Failed for\n{g}\ne={e}, n={n}, s={s}, ei={ei}, ni={ni}'
                                                    .format(**locals()))
                    else:
                        for si in xrange(SUE.shape[0]):
                            for ei, ni in product(SUE[si], SUN[si]):
                                if ei != ni and ei != -1 and ni != -1:
                                    self.assertTrue(ni in N[ei],
                                                    'Failed for\n{g}\ne={e}, n={n}, s={s}, ei={ei}, ni={ni}'
                                                    .format(**locals()))

    def test_neighbours_not_neighbour_of_itself(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
                    N = g.neighbours(e, e, s)
                    for ei, E in enumerate(N):
                        self.assertTrue(ei not in E,
                                        'Failed for\n{g}\ne={e}, s={s}, ei={ei}, E={E}'.format(**locals()))

    def test_boundary_mask_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.boundary_mask(-1)
            with self.assertRaises(AssertionError):
                g.boundary_mask(g.dim + 1)

    def test_boundary_mask_shape(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertEqual(g.boundary_mask(d).shape, (g.size(d),))

    def test_boundary_mask_dtype(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertEqual(g.boundary_mask(d).dtype, np.dtype('bool'),
                                 'Failed for\n{g}\nd={d}'.format(**locals()))

    def test_boundary_mask_entries_codim1(self):
        for g in self.grids:
            BM = g.boundary_mask(1)
            SE = g.superentities(1, 0)
            for ei, b in enumerate(BM):
                E = SE[ei]
                self.assertEqual(E[E > -1].size <= 1, b)

    def test_boundary_mask_entries_codim0(self):
        for g in self.grids:
            BM0 = g.boundary_mask(0)
            BM1 = g.boundary_mask(1)
            SE = g.subentities(0, 1)
            for ei, b in enumerate(BM0):
                S = SE[ei]
                self.assertEqual(np.any(BM1[S[S > -1]]), b)

    def test_boundary_mask_entries_codim_d(self):
        for g in self.grids:
            for d in xrange(2, g.dim + 1):
                BMD = g.boundary_mask(d)
                BM1 = g.boundary_mask(1)
                SE = g.superentities(d, 1)
                for ei, b in enumerate(BMD):
                    S = SE[ei]
                    self.assertEqual(np.any(BM1[S[S > -1]]), b)

    def test_boundaries_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.boundaries(-1)
            with self.assertRaises(AssertionError):
                g.boundaries(g.dim + 1)

    def test_boundaries_shape(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertEqual(len(g.boundaries(d).shape), 1)

    def test_boundaries_dtype(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertEqual(g.boundaries(d).dtype, np.dtype('int32'),
                                 'Failed for\n{g}\nd={d}'.format(**locals()))

    def test_boundaries_entry_value_range(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                np.testing.assert_array_less(g.boundaries(d), g.size(d))
                np.testing.assert_array_less(-1, g.boundaries(d))

    def test_boundaries_entries(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                np.testing.assert_array_equal(np.where(g.boundary_mask(d))[0], g.boundaries(d))


@GridSubclassForImplemetorsOf(AffineGridInterface)
class AffineGridTestInterface(GridClassTestInterface):

    def setUp(self):
        self.assertTrue(hasattr(self, 'grids'))
        self.assertGreater(len(self.grids), 0)

    def test_dim_outer(self):
        for g in self.grids:
            self.assertIsInstance(g.dim_outer, int)
            self.assertGreaterEqual(g.dim_outer, g.dim)

    def test_reference_element_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.reference_element(-1)
            with self.assertRaises(AssertionError):
                g.reference_element(g.dim + 1)

    def test_reference_element_type(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                self.assertIsInstance(g.reference_element(d), ReferenceElementInterface)

    def test_reference_element_transitivity(self):
        for g in self.grids:
            for d in xrange(1, g.dim + 1):
                self.assertIs(g.reference_element(d), g.reference_element(0).sub_reference_element(d))

    def test_embeddings_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.embeddings(-1)
            with self.assertRaises(AssertionError):
                g.embeddings(g.dim + 1)

    def test_embeddings_shape(self):
        for g in self.grids:
            for d in xrange(g.dim + 1):
                RES = g.embeddings(d)
                self.assertEqual(len(RES), 2)
                A, B = RES
                self.assertEqual(A.shape, (g.size(d), g.dim_outer, g.dim - d))
                self.assertEqual(B.shape, (g.size(d), g.dim_outer))

    def test_embeddings_transitivity(self):
        for g in self.grids:
            for d in xrange(1, g.dim + 1):
                AD1, BD1 = g.embeddings(d - 1)
                AD, BD = g.embeddings(d)
                SE = g.superentities(d, d - 1)
                SEI = g.superentity_indices(d, d - 1)
                ASUB, BSUB = g.reference_element(d - 1).subentity_embedding(1)
                for e in xrange(g.size(d)):
                    np.testing.assert_allclose(AD[e], np.dot(AD1[SE[e, 0]], ASUB[SEI[e, 0]]))
                    np.testing.assert_allclose(BD[e], np.dot(AD1[SE[e, 0]], BSUB[SEI[e, 0]]) + BD1[SE[e, 0]])

    def test_jacobian_inverse_transposed_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.jacobian_inverse_transposed(-1)
            with self.assertRaises(AssertionError):
                g.jacobian_inverse_transposed(g.dim + 1)
            with self.assertRaises(AssertionError):
                g.jacobian_inverse_transposed(g.dim)

    def test_jacobian_inverse_transposed_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                self.assertEqual(g.jacobian_inverse_transposed(d).shape, (g.size(d), g.dim_outer, g.dim - d))

    def test_jacobian_inverse_transposed_values(self):
        for g in self.grids:
            for d in xrange(g.dim):
                JIT = g.jacobian_inverse_transposed(d)
                A, _ = g.embeddings(d)
                for e in xrange(g.size(d)):
                    np.testing.assert_allclose(JIT[e], np.linalg.pinv(A[e]).T)

    def test_integration_elements_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.integration_elements(-1)
            with self.assertRaises(AssertionError):
                g.integration_elements(g.dim + 1)

    def test_integration_elements_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                self.assertEqual(g.integration_elements(d).shape, (g.size(d),))

    def test_integration_elements_values(self):
        for g in self.grids:
            for d in xrange(g.dim - 1):
                IE = g.integration_elements(d)
                A, _ = g.embeddings(d)
                for e in xrange(g.size(d)):
                    np.testing.assert_allclose(IE[e], np.sqrt(np.linalg.det(np.dot(A[e].T, A[e]))))
            np.testing.assert_allclose(g.integration_elements(g.dim), 1)

    def test_volumes_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.volumes(-1)
            with self.assertRaises(AssertionError):
                g.volumes(g.dim + 1)

    def test_volumes_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                self.assertEqual(g.volumes(d).shape, (g.size(d),))

    def test_volumes_values(self):
        for g in self.grids:
            for d in xrange(g.dim - 1):
                V = g.volumes(d)
                IE = g.integration_elements(d)
                np.testing.assert_allclose(V, IE * g.reference_element(d).volume)

    def test_volumes_inverse_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.volumes_inverse(-1)
            with self.assertRaises(AssertionError):
                g.volumes_inverse(g.dim + 1)

    def test_volumes_inverse_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                self.assertEqual(g.volumes_inverse(d).shape, (g.size(d),))

    def test_volumes_inverse_values(self):
        for g in self.grids:
            for d in xrange(g.dim - 1):
                VI = g.volumes_inverse(d)
                V = g.volumes(d)
                np.testing.assert_allclose(VI, np.reciprocal(V))

    def test_unit_outer_normals_shape(self):
        for g in self.grids:
            SE = g.subentities(0, 1)
            self.assertEqual(g.unit_outer_normals().shape, SE.shape + (g.dim_outer,))

    def test_unit_outer_normals_normed(self):
        for g in self.grids:
            UON = g.unit_outer_normals()
            np.testing.assert_allclose(np.sum(UON ** 2, axis=-1), 1)

    def test_unit_outer_normals_normal(self):
        for g in self.grids:
            SE = g.subentities(0, 1)
            A, _ = g.embeddings(1)
            SEE = A[SE, ...]
            UON = g.unit_outer_normals()
            np.testing.assert_allclose(np.sum(SEE * UON[..., np.newaxis], axis=-2), 0)

    def test_unit_outer_normals_neighbours(self):
        for g in self.grids:
            UON = g.unit_outer_normals()
            SE = g.superentities(1)
            SEI = g.superentity_indices(1)
            if SE.shape[1] < 2:
                continue
            for se, sei in izip(SE, SEI):
                if se[0] == -1 or se[1] == -1:
                    continue
                np.testing.assert_allclose(UON[se[0], sei[0]], -UON[se[1], sei[1]])

    def test_centers_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.centers(-1)
            with self.assertRaises(AssertionError):
                g.centers(g.dim + 1)

    def test_centers_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                self.assertEqual(g.centers(d).shape, (g.size(d), g.dim_outer))

    def test_centers_values(self):
        for g in self.grids:
            for d in xrange(g.dim):
                A, B = g.embeddings(d)
                np.testing.assert_allclose(g.centers(d), B + A.dot(g.reference_element(d).center()))

    def test_diameters_wrong_arguments(self):
        for g in self.grids:
            with self.assertRaises(AssertionError):
                g.diameters(-1)
            with self.assertRaises(AssertionError):
                g.diameters(g.dim + 1)

    def test_diameters_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                self.assertEqual(g.diameters(d).shape, (g.size(d),))

    def test_diameters_non_negative(self):
        for g in self.grids:
            for d in xrange(g.dim - 1):
                self.assertGreaterEqual(np.min(g.diameters(d)), 0)

    def test_diameters_values(self):
        for g in self.grids:
            for d in xrange(g.dim - 1):
                A, _ = g.embeddings(d)
                np.testing.assert_allclose(g.diameters(d), g.reference_element(d).mapped_diameter(A))

    def test_quadrature_points_wrong_arguments(self):
        for g in self.grids:
            for d in xrange(g.dim):
                with self.assertRaises(Exception):
                    g.quadrature_points(d, order=1, npoints=1)
                with self.assertRaises(Exception):
                    g.quadrature_points(d)
                os, ps = g.reference_element(d).quadrature_info()
                for t in os.keys():
                    with self.assertRaises(Exception):
                        g.quadrature_points(d, order=max(os[t]) + 1, quadrature_type=t)
                    with self.assertRaises(Exception):
                        g.quadrature_points(d, npoints=max(ps[t]) + 1, quadrature_type=t)

    def test_quadrature_points_shape(self):
        for g in self.grids:
            for d in xrange(g.dim):
                os, ps = g.reference_element(d).quadrature_info()
                for t in os.keys():
                    for o, p in izip(os[t], ps[t]):
                        self.assertEqual(g.quadrature_points(d, order=o, quadrature_type=t).shape,
                                         (g.size(d), p, g.dim_outer))
                        self.assertEqual(g.quadrature_points(d, npoints=p, quadrature_type=t).shape,
                                         (g.size(d), p, g.dim_outer))

    def test_quadrature_points_values(self):
        for g in self.grids:
            for d in xrange(g.dim):
                A, B = g.embeddings(d)
                os, ps = g.reference_element(d).quadrature_info()
                for t in os.keys():
                    for o, p in izip(os[t], ps[t]):
                        Q = g.quadrature_points(d, order=o, quadrature_type=t)
                        q, _ = g.reference_element(d).quadrature(order=o, quadrature_type=t)
                        np.testing.assert_allclose(Q, g.quadrature_points(d, npoints=p, quadrature_type=t))
                        np.testing.assert_allclose(Q, B[:, np.newaxis, :] + np.einsum('eij,qj->eqi', A, q))

# this needs to go into every module that wants to use dynamically generated types, ie. testcases, below the test code
from pymor.core.dynamic import *

if __name__ == "__main__":
    runmodule(name='pymortests.grid')
