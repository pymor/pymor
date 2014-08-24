# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import datetime
import sys
import logging
import timeit

from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid

import os


def initialize_logging():
    logger = logging.getLogger()
    logger.removeHandler(logger.handlers[0])
    logger.setLevel('INFO')

    hstdout = logging.StreamHandler(sys.stdout)
    fstdout = logging.Formatter('%(message)s')
    hstdout.setFormatter(fstdout)
    logger.addHandler(hstdout)

    # Output logging information to file
    t = datetime.datetime.now()
    output_dir = os.path.join(os.path.abspath(__file__).split('/src/')[0], 'logs')
    os.mkdir(output_dir)
    filename = os.path.join(output_dir,
                            'gridbenchmark-{}-{}-{}-{}{}{}.log'.format(t.year, t.month, t.day,
                                                                       t.hour, t.minute, t.second))
    hfile = logging.FileHandler(filename)
    ffile = logging.Formatter('%(asctime)s %(levelname)s\t%(message)s')
    hfile.setFormatter(ffile)
    logger.addHandler(hfile)

    print('Logging to {}\n'.format(filename))


def log(msg):
    logging.info(msg)


class GridBenchmark(object):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def make_instance(self):
        self.g = self.cls(*self.args, **self.kwargs)

    def setup(self):
        pass

    def run(self):
        pass

    def benchmark(self):
        return min(timeit.repeat(self.run, self.setup, number=1, repeat=3))


class Superentities10Benchmark(GridBenchmark):

    def setup(self):
        self.make_instance()

    def run(self):
        self.g.superentities(1, 0)


class SuperentityIndices10Benchmark(GridBenchmark):

    def setup(self):
        self.make_instance()
        self.g.superentities(1, 0)

    def run(self):
        self.g.superentity_indices(1, 0)


class Superentity10BothBenchmark(GridBenchmark):

    def setup(self):
        self.make_instance()

    def run(self):
        self.g.superentities(1, 0)
        self.g.superentity_indices(1, 0)

if __name__ == "__main__":
    initialize_logging()
    grid_sizes = [128, 256, 512]

    for c in (RectGrid, TriaGrid):
        log('Timing g.superentities(1, 0) for {}'.format(c.__name__))
        for n in grid_sizes:
            B = Superentities10Benchmark(c, (n, n))
            log('{2}(({0},{0})): {1}'.format(n, B.benchmark(), c.__name__))
        log('')

    for c in (RectGrid, TriaGrid):
        log('Timing g.superentities(1, 0) and g.superentity_indices(1, 0) for {}'.format(c.__name__))
        for n in grid_sizes:
            B = Superentity10BothBenchmark(c, (n, n))
            log('{2}(({0},{0})): {1}'.format(n, B.benchmark(), c.__name__))
        log('')
