# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import hashlib
import os
import sys
import numpy as np
from pickle import dump, load
from pkg_resources import resource_filename, resource_stream


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
