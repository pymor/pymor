# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
from scipy.io import loadmat, mmread
from scipy.sparse import issparse
import numpy as np
import tempfile
import os
from contextlib import contextmanager
import shutil

from pymor.core.logger import getLogger


def _loadmat(path, key=None):

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


def _mmread(path, key=None):

    if key:
        raise IOError('Cannot specify "key" for Matrix Market file')
    try:
        matrix = mmread(path)
        if issparse(matrix):
            matrix = matrix.tocsc()
        return matrix
    except Exception as e:
        raise IOError(e)


def _load(path, key=None):
    data = np.load(path)
    if isinstance(data, dict):
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


def _loadtxt(path, key=None):
    if key:
        raise IOError('Cannot specify "key" for TXT file')
    try:
        return np.loadtxt(path)
    except Exception as e:
        raise IOError(e)


def load_matrix(path, key=None):

    logger = getLogger('pymor.tools.io.load_matrix')
    logger.info('Loading matrix from file ' + path)

    path_parts = path.split('.')
    if len(path_parts[-1]) == 3:
        extension = path_parts[-1].lower()
    elif path_parts[-1].lower() == 'gz' and len(path_parts) >= 2 and len(path_parts[-2]) == 3:
        extension = '.'.join(path_parts[-2:]).lower()
    else:
        extension = None

    file_format_map = {'mat': ('MATLAB', _loadmat),
                       'mtx': ('Matrix Market', _mmread),
                       'mtz.gz': ('Matrix Market', _mmread),
                       'npy': ('NPY/NPZ', _load),
                       'npz': ('NPY/NPZ', _load),
                       'txt': ('Text', _loadtxt)}

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
    """Cross Platform safe equivalent of re-opening a NamedTemporaryFile
    Creates an automatically cleaned up temporary directory with a single file therein.

    name: filename component, defaults to 'temp_file'
    dir: the parent dir of the new tmp dir. defaults to tempfile.gettempdir()
    """
    parent_dir = parent_dir or tempfile.gettempdir()
    name = name or 'temp_file'
    dirname = tempfile.mkdtemp(dir=parent_dir)
    path = os.path.join(dirname, name)
    yield path
    shutil.rmtree(dirname)
