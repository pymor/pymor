from __future__ import absolute_import, division, print_function

from pymor import parameters
from pymortests.base import TestBase, runmodule


class TestCubicParameterspace(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.space = parameters.CubicParameterSpace({'diffusionl': 1}, 0.1, 1)
        self.samples = 100

    def _check_values(self, values):
        self.assertEqual(len(values), self.samples)
        for value in values:
            self.assertTrue(self.space.contains(value))

    def test_uniform(self):
        values = list(self.space.sample_uniformly(self.samples))
        self._check_values(values)

    def test_randomly(self):
        values = list(self.space.sample_randomly(self.samples))
        self._check_values(values)


if __name__ == "__main__":
    runmodule(name='pymortests.parameters')
