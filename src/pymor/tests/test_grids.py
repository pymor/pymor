import unittest

from pymor.grid.interfaces import IConformalTopologicalGrid, ISimpleAffineGrid
from pymor.grid.rect import Rect
from pymor.grid.tria import Tria
from pymor.core.exceptions import CodimError


class ConformalTopologicalGridTest(unittest.TestCase):

    grids = []

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


class TestRect(ConformalTopologicalGridTest):

    @classmethod
    def setUpClass(cls):
        cls.grids = [Rect((2,4)), Rect((1,1)), Rect((42,42)), Rect((100,100))]


class TestTria(ConformalTopologicalGridTest):

    @classmethod
    def setUpClass(cls):
        cls.grids = [Tria((2,4)), Tria((1,1)), Tria((42,42)), Tria((100,100))]
