# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.core.cache import CacheableObject
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction


class QuadBTReductor(CacheableObject):

    def __init__(self, s, Hs):
        assert isinstance(s, np.ndarray)
        assert np.all(s.imag > 0)
        assert np.all(s.real == 0)
        if hasattr(Hs, 'transfer_function'):
            Hs = Hs.transfer_function
        assert isinstance(Hs, TransferFunction)
        # assert isinstance(Hs, TransferFunction | np.ndarray | list)

        nodes_left = s[::2]
        nodes_right = s[1::2]

        # Close left and right nodes under complex conjugation.
        nodes_left  = np.hstack([nodes_left, nodes_left.conj()[::-1]])
        nodes_right = np.hstack([nodes_right, nodes_right.conj()[::-1]])

        # Weights.
        weights_right = np.hstack([[nodes_right[1] - nodes_right[0]], nodes_right[2:] - nodes_right[0:-2],
                [nodes_right[-1] - nodes_right[-2]]]) / 2
        weights_right = np.sqrt(1/(2*np.pi))*np.sqrt(np.abs(weights_right))
        weights_left = np.hstack([[nodes_left[1] - nodes_left[0]], nodes_left[2:] - nodes_left[0:-2],
                [nodes_left[-1] - nodes_left[-2]]]) / 2
        weights_left = np.sqrt(1/(2*np.pi))*np.sqrt(np.abs(weights_left))

        Hs_left = np.empty((len(nodes_left), Hs.dim_output, Hs.dim_input), dtype=s[0].dtype)
        for i, ss in enumerate(nodes_left):
            Hs_left[i] = Hs.eval_tf(ss)

        Hs_right = np.empty((len(nodes_right), Hs.dim_output, Hs.dim_input), dtype=s[0].dtype)
        for i, ss in enumerate(nodes_right):
            Hs_right[i] = Hs.eval_tf(ss)

        self.nodes_left, self.nodes_right, self.Hs_left, self.Hs_right, self.weights_left, self.weights_right = \
            nodes_left, nodes_right, Hs_left, Hs_right, weights_left, weights_right

        self.__auto_init(locals())

    def reduce(self, r=None):
        nodes_left, nodes_right, Hs_left, Hs_right, weights_left, weights_right = \
            self.nodes_left, self.nodes_right, self.Hs_left, self.Hs_right, self.weights_left, self.weights_right

        n_left, n_right = len(nodes_left), len(nodes_right)
        p, m = self.Hs.dim_output, self.Hs.dim_input

        EBAR = np.zeros((n_left*p, n_right*m), dtype=np.complex128)
        ABAR = np.zeros((n_left*p, n_right*m), dtype=np.complex128)
        BBAR = np.zeros((n_left*p, m),         dtype=np.complex128)
        CBAR = np.zeros((p,        n_right*m), dtype=np.complex128)

        for k in range(n_left):
            BBAR[k*p:(k+1)*p, :] = weights_left[k]  * Hs_left[k, :, :]
        for j in range(n_right):
            CBAR[:, j*m:(j+1)*m] = weights_right[j] * Hs_right[j, :, :]

        for k in range(n_left):
            for j in range(n_right):
                denom = nodes_left[k] - nodes_right[j]

                EBAR[k*p:(k+1)*p, j*m:(j+1)*m] \
                    = -weights_left[k]*weights_right[j] * (Hs_left[k, :, :] - Hs_right[j, :, :]) / denom

                # For ABAR.
                ABAR[k*p:(k+1)*p, j*m:(j+1)*m] \
                    = -weights_left[k]*weights_right[j] \
                        * (nodes_left[k]*Hs_left[k, :, :] - nodes_right[j]*Hs_right[j, :, :]) / denom

        Z, S, Y = spla.svd(EBAR)
        Y = Y.conj().T

        E = np.eye(r, r)
        Sr = np.diag(S[:r]**(-1/2))
        A = (Sr @ Z[:, :r].conj().T) @ ABAR @ (Y[:, :r] @ Sr)
        C = CBAR @ (Y[:, :r] @ Sr)
        B = (Sr @ Z[:, :r].conj().T) @ BBAR

        return LTIModel.from_matrices(A, B, C, D=None, E=E)
