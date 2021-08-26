# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

if __name__ == '__main__':
    import pytest
    import os
    import sys
    from pathlib import Path

    this_dir = Path(__file__).resolve().parent
    pymor_root_dir = (this_dir / '..' / '..').resolve()

    result_file_fn = pymor_root_dir / 'pytest.azure.success'
    try:
        os.unlink(result_file_fn)
    except FileNotFoundError:
        pass

    profile = os.environ.get("PYMOR_HYPOTHESIS_PROFILE", "ci")
    args = ["--junitxml=test_results.xml", f"--hypothesis-profile={profile}"] + sys.argv[1:]
    if pytest.main(args) == pytest.ExitCode.OK:
        with open(result_file_fn, 'wt') as result_file:
            result_file.write('True')
