# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymortests.base import runmodule
from pymor.discretizers.builtin import RectGrid, TriaGrid


def test_with_newtype():
    g = RectGrid(num_intervals=(99, 99))
    g2 = g.with_(new_type=TriaGrid, domain=([0, 0], [2, 2]))

    assert isinstance(g2, TriaGrid)
    assert g2.num_intervals == (99, 99)
    assert np.all(g2.domain == ([0, 0], [2, 2]))


if __name__ == "__main__":
    runmodule(filename=__file__)
