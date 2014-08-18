# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import time
import numpy as np
import functools

from pymor.core.logger import getLogger


class Timer(object):
    """You can use me as a context manager, plain instance or decorator to time execution
    of a code scope::

        with Timer() as timer:
            do_some_stuff()
            do more stuff()
        #outputs time in (s)

        ### OR ###

        @timing.Timer('name', logging.debug)
        def function(*args):
            do_stuff

        function(1)
        #outputs time in (s) to logging.debug

        ### OR ###

        timer = timing.Timer()
        timer.start()
        do_stuff()
        timer.stop()
        print(timer.dt)
    """

    def __init__(self, section, log=getLogger(__name__)):
        self._section = section
        self._log = log
        self._start = 0

    def start(self):
        self.dt = -1
        self._start = time.clock()

    def stop(self):
        self.dt = time.clock() - self._start

    def __enter__(self):
        self.start()

    def __exit__(self, type_, value, traceback):
        self.stop()
        self._log('Execution of %s took %f (s)', self._section, self.dt)

    def __call__(self, func):
        func.decorated = self

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return new_func


def busywait(amount):
    arr = np.arange(1000)
    for _ in xrange(amount):
        np.random.shuffle(arr)
