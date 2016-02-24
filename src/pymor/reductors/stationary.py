# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


def reduce_stationary_coercive(*args, **kwargs):
    """DEPRECATED! Renamed to pymor.reductors.coercive.reduce_coercive."""
    from warnings import warn
    warn('Renamed to pymor.reductors.coercive.reduce_coercive.\n' +
         'This function will be renamed in the next release of pyMOR.')
    from pymor.reductors.coercive import reduce_coercive
    return reduce_coercive(*args, **kwargs)
