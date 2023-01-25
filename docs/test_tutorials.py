# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import nbformat
import pytest
from myst_nb.core.read import read_myst_markdown_notebook
from pytest_notebook.nb_regression import NBRegressionFixture
from pytest_notebook.plugin import gather_config_options

from pymortests.base import runmodule

TUT_DIR = Path(os.path.dirname(__file__)).resolve().absolute() / 'source'
_exclude_files = []
EXCLUDE = [TUT_DIR / t for t in _exclude_files]
TUTORIALS = [t for t in TUT_DIR.glob('tutorial_*md') if t not in EXCLUDE]


class NBLaxFixture(NBRegressionFixture):
    """Same functionality as base class, but result comparison for regressions is skipped."""

    def check(self, path):
        return super().check(path=path, raise_errors=False)


@pytest.fixture()
def nb_lax(pytestconfig):
    kwargs, _ = gather_config_options(pytestconfig)
    return NBLaxFixture(**kwargs)


@pytest.fixture(params=TUTORIALS, ids=[t.name for t in TUTORIALS])
def converted_nb(request):
    filename = request.param
    nb = read_myst_markdown_notebook(filename.read_text('utf8'), path=filename)
    with NamedTemporaryFile(suffix='.ipynb', mode='w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        f.flush()
        yield f.name


def test_check(converted_nb, nb_lax):
    nb_lax.check(converted_nb)


if __name__ == '__main__':
    runmodule(filename=__file__)
