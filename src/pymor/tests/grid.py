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

    def test_subentitites(self):
        for g in self.grids:
            for e in xrange(g.dim + 1):
                for s in xrange(e, g.dim + 1):
                    self.assertEqual(g.subentities(e, s).shape[0], g.size(e))
                with self.assertRaises(AssertionError):
                    g.subentities(e, g.dim + 1)
                    g.subentities(e, e - 1)
                    g.subentities(e, 0)
            with self.assertRaises(AssertionError):
                g.subentities(g.dim + 1, 0)
                g.subentities(-1, 0)


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
