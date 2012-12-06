'''
Created on Nov 16, 2012

@author: r_milk01
'''
import numpy as np
import logging
import nose
import pprint
from pymor.grid.interfaces import IConformalTopologicalGrid, ISimpleAffineGrid
#mandatory so all Grid classes are created
from pymor.grid import *
import unittest

def make_testcase_classes(grid_types, TestCase):
    for GridType in grid_types:
        if GridType.has_interface_name():
            continue
        cname = '{}{}'.format(GridType.__name__, TestCase.__name__)
        #saves a new type called cname with correct bases and class dict in globals
        globals()[cname] = type(cname, (TestCase,), {'Grid': GridType})


class ConformalTopologicalGridTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grids = cls.Grid.test_instances() if cls.__name__ != 'ConformalTopologicalGridTest' else []

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

    def test_subentities_entry_value_range(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
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



class SimpleAffineGridTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grids = cls.Grid.test_instances() if cls.__name__ != 'SimpleAffineGridTest' else []

    def test_volumes(self):
        for grid in self.grids:
            dim = grid.dim
            for codim in range(dim+1):
                self.assertGreater(np.min(grid.volumes(codim)), 0)
                self.assertGreater(np.min(grid.volumes_inverse(codim)), 0)
                self.assertGreater(np.min(grid.diameters(codim)), 0)

make_testcase_classes(IConformalTopologicalGrid.implementors(True), ConformalTopologicalGridTest)
make_testcase_classes(ISimpleAffineGrid.implementors(True), SimpleAffineGridTest)

#this hard fails module import atm
#make_testcase_classes(IConformalTopologicalGrid.implementors(True), ConformalTopologicalGridTest)

if __name__ == "__main__":
#    nose.core.runmodule(name='__main__')
    logging.basicConfig(level=logging.INFO)
    unittest.main()
