# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import os
import sys
from pathlib import Path
import io
import importlib.machinery
import importlib.util

from docutils.core import publish_doctree
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import flag, register_directive
import pytest

from pymor.tools.io import change_to_directory
from pymortests.base import runmodule
from pymortests.demos import _test_demo

TUT_DIR = Path(os.path.dirname(__file__)).resolve() / 'source'
_exclude_files = []
EXCLUDE = [TUT_DIR / t for t in _exclude_files]
TUTORIALS = [t for t in TUT_DIR.glob('tutorial_*rst') if t not in EXCLUDE]
TUTORIALS += [t for t in TUT_DIR.glob('tutorial_*md') if t not in EXCLUDE]


class CodeCell(Directive):

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'hide-output': flag,
                   'hide-code': flag,
                   'raises': flag}
    has_content = True

    def run(self):
        self.assert_has_content()
        if 'raises' in self.options:
            text = 'try:\n    ' + '\n    '.join(
                self.content) + '\nexcept:\n    import traceback; traceback.print_exc()'
        else:
            text = '\n'.join(self.content)
        print('# %%')
        print(text)
        print()
        return []


@pytest.fixture(params=TUTORIALS, ids=[t.name for t in TUTORIALS])
def tutorial_code(request):
    filename = request.param
    with change_to_directory(TUT_DIR):
        code = io.StringIO()
        register_directive('jupyter-execute', CodeCell)
        with open(filename, 'rt') as f:
            original = sys.stdout
            sys.stdout = code
            publish_doctree(f.read(), settings_overrides={'report_level': 42})
            sys.stdout = original
    code.seek(0)
    source_fn = Path(f'{str(filename).replace(".rst", "_rst")}_extracted.py')
    with open(source_fn, 'wt') as source:
        # filter line magics
        source.write(''.join([line for line in code.readlines() if not line.startswith('%')]))
    return request.param, source_fn


def test_tutorial(tutorial_code):
    filename, source_module_path = tutorial_code

    # make sure (picture) resources can be loaded as in sphinx-build
    with change_to_directory(TUT_DIR):
        def _run():
            loader = importlib.machinery.SourceFileLoader(source_module_path.stem, str(source_module_path))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
        try:
            # wrap module execution in hacks to auto-close Qt-Apps, etc.
            _test_demo(_run)
        except Exception as e:
            print(f'Failed: {source_module_path}')
            raise e


if __name__ == "__main__":
    runmodule(filename=__file__)
