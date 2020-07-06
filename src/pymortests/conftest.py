# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
from hypothesis import settings, Verbosity, HealthCheck

_common_settings = {"print_blob": True, "suppress_health_check": (HealthCheck.too_slow, HealthCheck.data_too_large,),
                    "deadline": 1000, "verbosity": Verbosity.normal}
settings.register_profile("ci_large", max_examples=400, **_common_settings)
settings.register_profile("ci_pr", max_examples=100, **_common_settings)
settings.register_profile("ci", max_examples=25, **_common_settings)
settings.register_profile("dev", max_examples=10, **_common_settings)
_common_settings["verbosity"] = Verbosity.verbose
settings.register_profile("debug", max_examples=10, **_common_settings)
settings.load_profile(os.getenv(u'PYMOR_HYPOTHESIS_PROFILE', 'dev'))

