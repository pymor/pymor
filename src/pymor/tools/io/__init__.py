# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import shutil
import tempfile
from contextlib import contextmanager

from .matrices import load_matrix, save_matrix
from pymor.tools.deprecated import Deprecated


@Deprecated('pymor.tools.io.vtk.read_vtkfile')
def read_vtkfile(*args, **kwargs):
    from .vtk import read_vtkfile
    return read_vtkfile(*args, **kwargs)


@Deprecated('pymor.tools.io.vtk.write_vtk_collection')
def write_vtk_collection(*args, **kwargs):
    from .vtk import write_vtk_collection
    return write_vtk_collection(*args, **kwargs)


@contextmanager
def safe_temporary_filename(name=None, parent_dir=None):
    """Cross~platform safe equivalent of re-opening a NamedTemporaryFile.

    Creates an automatically cleaned up temporary directory with a single file therein.

    Parameters
    ----------
    name
        Filename component, defaults to 'temp_file'.
    dir
        The parent dir of the new temporary directory.
        Defaults to tempfile.gettempdir().
    """
    parent_dir = parent_dir or tempfile.gettempdir()
    name = name or 'temp_file'
    dirname = tempfile.mkdtemp(dir=parent_dir)
    path = os.path.join(dirname, name)
    yield path
    shutil.rmtree(dirname)


@contextmanager
def change_to_directory(name):
    """Change current working directory to `name` for the scope of the context."""
    old_cwd = os.getcwd()
    try:
        yield os.chdir(name)
    finally:
        os.chdir(old_cwd)


def file_owned_by_current_user(filename):
    try:
        return os.stat(filename).st_uid == os.getuid()
    except AttributeError:
        # this is actually less secure than above since getuser looks in env for username
        # a portable way to getuid might be in psutil
        from getpass import getuser
        import win32security
        f = win32security.GetFileSecurity(filename, win32security.OWNER_SECURITY_INFORMATION)
        username, _, _ = win32security.LookupAccountSid(None, f.GetSecurityDescriptorOwner())
        return username == getuser()
