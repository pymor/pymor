# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import hashlib
import os
import sys
from pprint import pformat
from functools import wraps

import hypothesis
import numpy as np
from pickle import dump, load
from pkg_resources import resource_filename, resource_stream
from pytest import skip

from pymor.algorithms.basic import almost_equal, relative_error
from pymor.core import config
from pymor.core.exceptions import DependencyMissing


def runmodule(filename):
    import pytest

    sys.exit(pytest.main(sys.argv[1:] + [filename]))


def check_results(test_name, params, results, *args):
    params = str(params)
    tols = (1e-13, 1e-13)
    keys = {}
    for arg in args:
        if isinstance(arg, tuple):
            assert len(arg) == 2
            tols = arg
        else:
            keys[arg] = tols

    assert results is not None
    assert set(keys.keys()) <= set(results.keys()), \
        f'Keys {set(keys.keys()) - set(results.keys())} missing in results dict'
    results = {k: np.asarray(results[k]) for k in keys.keys()}
    assert all(v.dtype != object for v in results.values())

    basepath = resource_filename('pymortests', 'testdata/check_results')
    arg_id = hashlib.sha1(params.encode()).hexdigest()
    filename = resource_filename('pymortests', f'testdata/check_results/{test_name}/{arg_id}')
    testname_dir = os.path.join(basepath, test_name)

    def _dump_results(fn, res):
        with open(fn, 'wb') as f:
            f.write((params + '\n').encode())
            res = {k: v.tolist() for k, v in res.items()}
            dump(res, f, protocol=2)

    try:
        with resource_stream('pymortests', f'testdata/check_results/{test_name}/{arg_id}') as f:
            f.readline()
            old_results = load(f)
    except FileNotFoundError:
        if not os.path.exists(testname_dir):
            os.mkdir(testname_dir)
        _dump_results(filename, results)
        assert False, \
            f'No results found for test {test_name} ({params}), saved current results. Remember to check in {filename}.'

    for k, (atol, rtol) in keys.items():
        if not np.all(np.allclose(old_results[k], results[k], atol=atol, rtol=rtol)):
            abs_errs = np.abs(results[k] - old_results[k])
            rel_errs = abs_errs / np.abs(old_results[k])
            _dump_results(filename + '_changed', results)
            assert False, (f'Results for test {test_name}({params}, key: {k}) have changed.\n'
                           f'(maximum error: {np.max(abs_errs)} abs / {np.max(rel_errs)} rel).\n'
                           f'Saved new results in {filename}_changed')


def assert_all_almost_equal(U, V, product=None, sup_norm=False, rtol=1e-14, atol=1e-14):
    cmp_array = almost_equal(U, V, product=product, sup_norm=sup_norm, rtol=rtol, atol=atol)
    too_large_relative_errors = dict((i, relative_error(u, v, product=product))
                                     for i, (u, v, f) in enumerate(zip(U, V, cmp_array)) if not f)
    assert np.all(cmp_array), f'Relative errors for not-equal elements:{pformat(too_large_relative_errors)}'


def might_exceed_deadline(deadline=-1):
    """For hypothesis magic to work properly this must be the topmost decorator on test function"""
    def _outer_wrapper(func):
        @wraps(func)
        def _inner_wrapper(*args, **kwargs):
            dl = deadline
            if os.environ.get('PYMOR_ALLOW_DEADLINE_EXCESS', False):
                dl = None
            elif dl == -1:
                dl = hypothesis.settings.default.deadline.total_seconds() * 1e3
            assert dl is None or dl > 1
            return hypothesis.settings(deadline=dl)(func)(*args, **kwargs)
        return _inner_wrapper
    return _outer_wrapper


def skip_if_missing(config_name):
    """Wrapper for requiring certain module dependencies on tests.

    This will explicitly call :meth:`pymor.core.config.Config.require` with
    given `config_name`, so it's usable even in places where a pyMOR module
    that requires that package is not imported (yet).

    Parameters
    ----------
    config_name
        if `pymor.core.config.HAVE_config_name` evaluates to `False` and we're not in the
        Docker CI environment, the test is skipped.
    """
    def _outer_wrapper(func):
        @wraps(func)
        def _inner_wrapper(*args, **kwargs):
            try:
                config.config.require(config_name)
            except DependencyMissing as dm:
                # skip does not return
                if config_name in str(dm.dependency) and not os.environ.get('DOCKER_PYMOR', False):
                    skip_string = 'skipped test due to missing dependency ' + config_name
                    skip(skip_string)
                raise dm
            func(*args, **kwargs)
        return _inner_wrapper
    return _outer_wrapper
