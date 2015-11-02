# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.discretizations.iosys as iosys

def test_cgf():
    n = 100
    m = 2
    p = 3
    np.random.seed(1)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    lti = iosys.LTISystem.from_matrices(A, B, C)

    lti.compute_cgf()
    AZZT = lti.A.apply(lti._cgf).data.T.dot(lti._cgf.data)
    BBT = lti.B.data.T.dot(lti.B.data)

    assert np.linalg.norm(AZZT + AZZT.T + BBT) / np.linalg.norm(BBT) < 1e-10

def test_ogf():
    n = 100
    m = 2
    p = 3
    np.random.seed(1)
    A = np.random.randn(n, n) - n * np.eye(n)
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    lti = iosys.LTISystem.from_matrices(A, B, C)

    lti.compute_ogf()
    ATZZT = lti.A.apply_adjoint(lti._ogf).data.T.dot(lti._ogf.data)
    CTC = lti.C.data.T.dot(lti.C.data)

    assert np.linalg.norm(ATZZT + ATZZT.T + CTC) / np.linalg.norm(CTC) < 1e-10

if __name__ == "__main__":
    runmodule(filename=__file__)
