#!/usr/bin/env python
import numpy as np

from pymor.core import interfaces


class Interface(interfaces.BasicInterface):

    id = 'common.discretefunction.continuous'
    name = id
    dim_domain = -1
    dim_range = -1
    order = -1
    size = -1

    @interfaces.abstractmethod
    def points(self):
        pass

    @interfaces.abstractmethod
    def values(self):
        pass

    def visualize(self):
        import matplotlib.pyplot as plt
        plt.plot(self.points(), self.values(), label=self.name)
        plt.legend()
        plt.show()


class P1(Interface):

    id = Interface.id + '.p1'
    dim_domain = 1
    dim_range = 1
    order = 1

    def __init__(self, grid, vector, name=id):
        assert grid.dim == 1
        self.grid = grid
        vector = np.array(vector, copy=False, ndmin=1)
        assert vector.ndim == 1 or (vector.ndim == 2
                                    and (vector.shape[0] == 1 or vector.shape[1] == 1))
        assert vector.size == self.grid.size(1)
        self.size == vector.size
        self.vector = vector
        self.name = name

    def points(self):
        return self.grid.centers(1)

    def values(self):
        return self.vector


if __name__ == '__main__':
    import pymor.grid.oned as grid
    grid = grid.Oned([0., 1.], 8)
    vector = grid.centers(1)
    df = P1(grid, vector)
    df.visualize()
