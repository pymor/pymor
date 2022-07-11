# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
from functools import wraps

from hypothesis import settings, Verbosity, HealthCheck
import pytest
import numpy as np

_common_settings = {
    "print_blob": True,
    "suppress_health_check": (HealthCheck.data_too_large, HealthCheck.too_slow),
    "deadline": 1000,
    "verbosity": Verbosity.normal,
}
settings.register_profile("ci_large", max_examples=400, **_common_settings)
settings.register_profile("ci_pr", max_examples=80, **_common_settings)
settings.register_profile("ci", max_examples=25, **_common_settings)
settings.register_profile("dev", max_examples=10, **_common_settings)
_common_settings["verbosity"] = Verbosity.verbose
settings.register_profile("debug", max_examples=10, **_common_settings)
settings.load_profile(os.getenv(u'PYMOR_HYPOTHESIS_PROFILE', 'dev'))

""" This makes sure all our fixtures are available to all tests

Individual test modules MUST NOT import fixtures from `pymortests.fixtures`,
as this can have strange side effects.
"""
pytest_plugins = [
    "pymortests.fixtures.analyticalproblem",
    "pymortests.fixtures.function",
    "pymortests.fixtures.grid",
    "pymortests.fixtures.model",
    "pymortests.fixtures.operator",
    "pymortests.fixtures.parameter",
]


@pytest.fixture(autouse=True)
def monkey_np_testing(monkeypatch):
    """All tests automagically use this, we only change the default tolerances

    monkey np.testing.assert_allclose to behave the same as np.allclose
    for some reason, the default atol of np.testing.assert_allclose is 0
    while it is 1e-8 for np.allclose
    """
    real_all_close = np.testing.assert_allclose

    @wraps(real_all_close)
    def monkey_allclose(a, b, rtol=1.e-5, atol=1.e-8):
        __tracebackhide__ = True  # Hide traceback for py.test
        return real_all_close(a, b, rtol=rtol, atol=atol)

    monkeypatch.setattr(np.testing, 'assert_allclose', monkey_allclose)
