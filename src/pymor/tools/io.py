# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from scipy.io import loadmat, mmread
from scipy.sparse import issparse
import numpy as np

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
            raise IOError('"{}" not found in MATLAB file {}'.format(key, path))

    data = [v for v in data.values() if isinstance(v, np.ndarray) or issparse(v)]

    if len(data) == 0:
        raise IOError('No matrix data contained in MATLAB file {}'.format(path))
    elif len(data) > 1:
        raise IOError('More than one matrix object stored in MATLAB file {}'.format(path))
    else:
        return data[0]


def _mmread(path, key=None):

    if key:
        raise IOError('Cannot specify "key" for Matrix Market file')
    try:
        matrix = mmread(path)
        return matrix.tocsc()
    except Exception as e:
        raise IOError(e)


def _load(path, key=None):
    data = np.load(path)
    if isinstance(data, dict):
        if key:
            try:
                matrix = data[key]
            except KeyError:
                raise IOError('"{}" not found in NPY file {}'.format(key, path))
        elif len(data) == 0:
            raise IOError('No data contained in NPY file {}'.format(path))
        elif len(data) > 1:
            raise IOError('More than one object stored in NPY file {}'.format(key, path))
        else:
            matrix = next(iter(data.values()))
    else:
        matrix = data
    if not isinstance(matrix, np.ndarray) and not issparse(matrix):
        raise IOError('Loaded data is not a matrix in NPY file {}').format(path)
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

    logger.warn('Could not detect file format. Trying all loaders ...')

    loaders = [_loadmat, _mmread, _loadtxt, _load]
    for loader in loaders:
        try:
            return loader(path, key)
        except IOError:
            pass

    raise IOError('Could not load file {} (key = {})'.format(path, key))
