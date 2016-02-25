# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


def cat_arrays(vector_arrays):
    """Return a new |VectorArray| which a concatenation of the arrays in `vector_arrays`."""
    vector_arrays = list(vector_arrays)
    total_length = sum(map(len, vector_arrays))
    cated_arrays = vector_arrays[0].empty(reserve=total_length)
    for a in vector_arrays:
        cated_arrays.append(a)
    return cated_arrays
