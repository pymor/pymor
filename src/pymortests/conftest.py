# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import pytest
import time
import os
from csv import DictWriter
from hypothesis import settings, Verbosity, HealthCheck

_common_settings = {"print_blob": True, "suppress_health_check": (HealthCheck.too_slow, HealthCheck.data_too_large,),
                    "deadline": 1000, "verbosity": Verbosity.verbose}
settings.register_profile("ci_large", max_examples=5000, **_common_settings)
settings.register_profile("ci", max_examples=100, **_common_settings)
settings.register_profile("dev", max_examples=10, **_common_settings)
settings.register_profile("debug", max_examples=10, **_common_settings)
settings.load_profile(os.getenv(u'PYMOR_HYPOTHESIS_PROFILE', 'dev'))

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
        started = time.process_time()
        yield
        self._elapsed_times[item.name] = started

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item):
        stopped = time.process_time()
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


def pytest_report_header(config):
    split = os.environ.get('PYMOR_TEST_HALF', None)
    if split is None:
        return None
    return "Test suite splitting in effect"


def pytest_collection_modifyitems(config, items):
    split = os.environ.get('PYMOR_TEST_HALF', None)
    if split is None:
        return
    skip = pytest.mark.skip(reason="Suite splitting in effect")
    testcount = len(items)
    if split == 'TOP':
        for item in items[:testcount//2]:
            item.add_marker(skip)
    else:
        for item in items[testcount//2:]:
            item.add_marker(skip)
