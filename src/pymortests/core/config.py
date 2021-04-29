# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


from pymor.core.config import config, _PACKAGES


def test_repr():
    repr(config)


def test_entries():
    for p in _PACKAGES:
        assert hasattr(config, 'HAVE_' + p)
        assert type(getattr(config, 'HAVE_' + p)) is bool
        getattr(config, p + '_VERSION')


def test_dir():
    d = dir(config)
    for p in _PACKAGES:
        assert 'HAVE_' + p in d
        assert p + '_VERSION' in d
