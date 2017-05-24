# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import pytest
import time
from csv import DictWriter

try:
    take_time = time.process_time
except AttributeError:
    # py < 2.7 fallback
    take_time = time.time


class ExecutionTimeCSV:

    def __init__(self):
        import pymor
        self._elapsed_times = {'version': pymor.__version__}

    @pytest.hookimpl()
    def pytest_addoption(self, parser):
        # this doesn't actually work atm
        parser.addoption('--timings-file', type=str, dest='timings-file', default='test_timings.csv')

    @pytest.hookimpl()
    def pytest_configure(self, config):
        self._filename = config.getoption('timings-file')

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        started = take_time()
        yield
        self._elapsed_times[item.name] = started

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item):
        stopped = take_time()
        yield
        self._elapsed_times[item.name] = str(stopped - self._elapsed_times[item.name])

    @pytest.hookimpl()
    def pytest_terminal_summary(self, *args, **kwargs):
        with open(self._filename, 'w') as csvfile:
            writer = DictWriter(csvfile, fieldnames=self._elapsed_times.keys())
            writer.writeheader()
            writer.writerow(self._elapsed_times)


def pytest_configure(config):
    config.pluginmanager.register(ExecutionTimeCSV(), 'ExecutionTimeCSV')
