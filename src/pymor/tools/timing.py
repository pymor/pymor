# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
import numpy as np
import functools

from pymor.core.logger import getLogger


class Timer:
    """Class for finding code runtime.

    You can use me as a context manager, plain instance or decorator to time execution
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
        self._start = time.perf_counter()

    def stop(self):
        self.dt = time.perf_counter() - self._start

    def __enter__(self):
        self.start()

    def __exit__(self, type_, value, traceback):
        self.stop()
        self._log.info('Execution of %s took %f (s)', self._section, self.dt)

    def __call__(self, func):
        func.decorated = self

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return new_func


def busywait(amount):
    arr = np.arange(1000)
    for _ in range(amount):
        np.random.shuffle(arr)
