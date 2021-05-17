# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import loadmat, mmread
from scipy.sparse import issparse, spmatrix

from pymor.core.logger import getLogger

MatrixType = Union[ArrayLike, spmatrix]


def _loadmat(path: Path, key: Optional[str] = None) -> MatrixType:
    try:
        data = loadmat(path, mat_dtype=True)
    except Exception as e:
        raise IOError(e)

    if key:
        try:
            return data[key]
        except KeyError:
            raise IOError(f'"{key}" not found in MATLAB file {path}')

    data = [v for v in data.values() if isinstance(v, np.ndarray) or issparse(v)]

    if len(data) == 0:
        raise IOError(f'No matrix data contained in MATLAB file {path}')
    elif len(data) > 1:
        raise IOError(f'More than one matrix object stored in MATLAB file {path}')
    else:
        return data[0]


def _mmread(path: Path, key: Optional[str] = None) -> MatrixType:
    if key:
        raise IOError('Cannot specify "key" for Matrix Market file')
    try:
        matrix = mmread(path)
        if issparse(matrix):
            matrix = matrix.tocsc()
        return matrix
    except Exception as e:
        raise IOError(e)


def _load(path: Path, key: Optional[str] = None) -> MatrixType:
    try:
        data = np.load(path)
    except Exception as e:
        raise IOError(e)
    if isinstance(data, (dict, np.lib.npyio.NpzFile)):
        if key:
            try:
                matrix = data[key]
            except KeyError:
                raise IOError(f'"{key}" not found in NPY file {path}')
        elif len(data) == 0:
            raise IOError(f'No data contained in NPY file {path}')
        elif len(data) > 1:
            raise IOError(f'More than one object stored in NPY file {path} for key {key}')
        else:
            matrix = next(iter(data.values()))
    else:
        matrix = data
    if not isinstance(matrix, np.ndarray) and not issparse(matrix):
        raise IOError(f'Loaded data is not a matrix in NPY file {path}')
    return matrix


def _loadtxt(path: Path, key: Optional[str] = None) -> MatrixType:
    if key:
        raise IOError('Cannot specify "key" for TXT file')
    try:
        return np.loadtxt(path)
    except Exception as e:
        raise IOError(e)


def _get_file_extension(path: Path) -> str:
    suffix_count = len(path.suffixes)
    if suffix_count and len(path.suffixes[-1]) == 4:
        extension = path.suffixes[-1].lower()
    elif path.suffixes[-1].lower() == '.gz' and suffix_count >= 2 and len(path.suffixes[-2]) == 4:
        extension = '.'.join(path.suffixes[-2:]).lower()
    else:
        extension = ''
    return extension


def load_matrix(path: Union[str, Path], key: Optional[str] = None) -> Union[ArrayLike, spmatrix]:
    """Load matrix from file.

    Parameters
    ----------
    path
        Path to the file.
    key
        Key of the matrix (only for NPY, NPZ, and MATLAB files).

    Returns
    -------
    matrix
        |NumPy array| of |SciPy spmatrix|.

    Raises
    ------
    IOError
        If loading fails.
    """
    logger = getLogger('pymor.tools.io.load_matrix')
    logger.info('Loading matrix from file %s', path)

    # convert if path is str
    path = Path(path)
    extension = _get_file_extension(path)

    file_format_map = {
        '.mat': ('MATLAB', _loadmat),
        '.mtx': ('Matrix Market', _mmread),
        '.mtz.gz': ('Matrix Market', _mmread),
        '.npy': ('NPY/NPZ', _load),
        '.npz': ('NPY/NPZ', _load),
        '.txt': ('Text', _loadtxt),
    }

    if extension in file_format_map:
        file_type, loader = file_format_map[extension]
        logger.info(file_type + ' file detected.')
        return loader(path, key)

    logger.warning('Could not detect file format. Trying all loaders ...')

    loaders = [_loadmat, _mmread, _loadtxt, _load]
    for loader in loaders:
        try:
            return loader(path, key)
        except IOError:
            pass

    raise IOError(f'Could not load file {path} (key = {key})')


@contextmanager
def SafeTemporaryFileName(name=None, parent_dir=None):
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
