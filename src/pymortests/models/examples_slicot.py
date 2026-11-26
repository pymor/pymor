# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Tests for :func:`pymor.models.examples.slicot_model`.

The regular tests do not access the SLICOT website for every test run. Instead, the benchmark
archives are kept in a local mirror directory, which is served by a local HTTP server, such that
the actual download code path (pooch, hash check, unzipping) is still exercised.

The mirror directory is given by the environment variable `PYMOR_SLICOT_MIRROR` (default: an OS
dependent cache directory). Missing archives are fetched from the SLICOT website once; if this is
not possible, the corresponding tests are skipped. In CI, the mirror directory is kept in the
GitLab CI cache.

`test_slicot_model_upstream` checks the archives on the SLICOT website itself and only runs when
the environment variable `PYMOR_TEST_SLICOT_UPSTREAM` is set (weekly scheduled CI job).
"""

import os
import threading
import warnings
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import dia_array

from pymor.models import examples
from pymor.models.iosys import LTIModel

pytestmark = pytest.mark.builtin

# name: (order, dim_input, dim_output, sampling_time)
EXPECTED = {
    'CDplayer': (120, 2, 2, 0),
    'MNA_1': (578, 9, 9, 0),
    'MNA_2': (9223, 18, 18, 0),
    'MNA_3': (4863, 22, 22, 0),
    'MNA_4': (980, 4, 4, 0),
    'MNA_5': (10913, 9, 9, 0),
    'beam': (348, 1, 1, 0),
    'build': (48, 1, 1, 0),
    'eady': (598, 1, 1, 0),
    'fom': (1006, 1, 1, 0),
    'heat-cont': (200, 1, 1, 0),
    'heat-disc': (200, 1, 1, 1),
    'iss': (270, 3, 3, 0),
    'pde': (84, 1, 1, 0),
    'peec': (480, 1, 1, 0),
    'random': (200, 1, 1, 0),
    'tline': (256, 2, 2, 0),
}


def _benchmark(name):
    return examples._SLICOT_BENCHMARKS[name.upper()]


def _check_model(name):
    import pooch

    # use a fresh pooch cache, such that the archive is actually downloaded
    with TemporaryDirectory() as cache_dir, \
            patch.object(pooch, 'retrieve', partial(pooch.retrieve, path=cache_dir)):
        m = examples.slicot_model(name)

    assert isinstance(m, LTIModel)
    assert m.name == 'SLICOT_' + name.upper()
    assert (m.order, m.dim_input, m.dim_output, m.sampling_time) == EXPECTED[name]
    for op in (m.A, m.B, m.C, m.D, m.E):
        if hasattr(op, 'matrix'):
            assert op.matrix.dtype == np.float64
            data = op.matrix.data if hasattr(op.matrix, 'tocsc') else op.matrix
            assert np.all(np.isfinite(data))
    if name in ('beam', 'build'):
        assert isinstance(m.A.matrix, np.ndarray)
    if name == 'tline':
        assert isinstance(m.A.matrix, dia_array)


def test_model_list_consistent():
    assert sorted(examples.slicot_model_names()) == sorted(EXPECTED)


def test_unknown_model():
    with pytest.raises(ValueError, match='Unknown SLICOT benchmark model'):
        examples.slicot_model('no-such-model')


@pytest.fixture(scope='session')
def slicot_mirror():
    """Local mirror of the SLICOT benchmark archives.

    Archives which are missing in the mirror directory (or have a wrong hash) are fetched from
    the SLICOT website, issuing a warning. If this fails for any archive, all tests using the
    mirror are skipped.
    """
    pooch = pytest.importorskip('pooch')
    path = Path(os.environ.get('PYMOR_SLICOT_MIRROR') or pooch.os_cache('pymor-slicot-mirror'))
    path.mkdir(parents=True, exist_ok=True)

    missing = {filename: md5 for filename, md5 in examples._SLICOT_BENCHMARKS.values()
               if not (path / f'{filename}.zip').is_file()
               or pooch.file_hash(path / f'{filename}.zip', alg='md5') != md5}
    if not missing:
        return path

    warnings.warn(f'SLICOT mirror {path} is incomplete, downloading {len(missing)} archive(s) from the '
                  f'SLICOT website: {", ".join(missing)}')
    try:
        for filename, md5 in missing.items():
            pooch.retrieve(f'{examples._SLICOT_BENCHMARK_BASE_URL}/{filename}.zip', known_hash=f'md5:{md5}',
                           fname=f'{filename}.zip', path=path)
    except (OSError, ValueError) as e:  # network errors are OSErrors, hash mismatches ValueErrors
        pytest.skip(f'cannot populate SLICOT mirror: {e}')

    return path


@pytest.fixture(scope='session')
def slicot_server(slicot_mirror):
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=slicot_mirror))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_slicot_model(name, slicot_server, monkeypatch):
    monkeypatch.setattr(examples, '_SLICOT_BENCHMARK_BASE_URL', slicot_server)
    for var in ('NO_PROXY', 'no_proxy'):  # do not route requests to the local server via a proxy
        monkeypatch.setenv(var, ','.join(filter(None, [os.environ.get(var), '127.0.0.1'])))
    _check_model(name)


@pytest.mark.skipif(not os.environ.get('PYMOR_TEST_SLICOT_UPSTREAM'),
                    reason='set PYMOR_TEST_SLICOT_UPSTREAM to check the data on the SLICOT website')
@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_slicot_model_upstream(name):
    _check_model(name)
