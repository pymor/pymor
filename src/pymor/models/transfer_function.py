# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import CacheableObject
from pymor.core.cache import cached
from pymor.operators.block import BlockOperator, BlockRowOperator, BlockColumnOperator, BlockDiagonalOperator
from pymor.parameters.base import ParametricObject, Mu


class TransferFunction(CacheableObject, ParametricObject):
    r"""Class for systems represented by a transfer function.

    This class describes input-output systems given by a (parametrized) transfer
    function :math:`H(s, \mu)`.

    Parameters
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    tf
        The transfer function H, given by a callable that takes a complex value `s` and,
        if parametric, a |parameter value| `mu`.
        The result of `tf(s, mu)` is a |NumPy array| of shape `(dim_output, dim_input)`.
    dtf
        The complex derivative of `H` with respect to `s` (optional).
    parameters
        The |Parameters| of the transfer function.
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
    name
        Name of the system.

    Attributes
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    tf
        The transfer function.
    dtf
        The complex derivative of the transfer function.
    """

    cache_region = 'memory'

    def __init__(self, dim_input, dim_output, tf, dtf=None, parameters={}, sampling_time=0, name=None):
        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        self.parameters_own = parameters
        self.__auto_init(locals())

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of inputs:  {self.dim_input}\n'
            f'    number of outputs: {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time\n'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}\n'
        return string

    @cached
    def eval_tf(self, s, mu=None):
        """Evaluate the transfer function.

        Parameters
        ----------
        s
            Laplace variable as a complex number.
        mu
            |Parameter values|.

        Returns
        -------
        Transfer function value as a 2D |NumPy array|.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if not self.parametric:
            return self.tf(s)
        else:
            return self.tf(s, mu=mu)

    @cached
    def eval_dtf(self, s, mu=None):
        """Evaluate the derivative of the transfer function.

        Parameters
        ----------
        s
            Laplace variable as a complex number.
        mu
            |Parameter values|.

        Returns
        -------
        Transfer function value as a 2D |NumPy array|.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if self.dtf is None:
            raise ValueError('The derivative was not given.')
        if not self.parametric:
            return self.dtf(s)
        else:
            return self.dtf(s, mu=mu)

    def freq_resp(self, w, mu=None):
        """Evaluate the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        """
        if self.sampling_time > 0 and not all(-np.pi <= wi <= np.pi for wi in w):
            self.logger.warning('Some frequencies are not in the [-pi, pi] interval.')
        w = 1j * w if self.sampling_time == 0 else np.exp(1j * w)
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        return np.stack([self.eval_tf(wi, mu=mu) for wi in w])

    def bode(self, w, mu=None):
        """Compute magnitudes and phases.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.

        Returns
        -------
        mag
            Transfer function magnitudes at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        phase
            Transfer function phases (in radians) at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        """
        w = np.asarray(w)
        mag = np.abs(self.freq_resp(w, mu=mu))
        phase = np.angle(self.freq_resp(w, mu=mu))
        phase = np.unwrap(phase, axis=0)
        return mag, phase

    def bode_plot(self, w, mu=None, ax=None, Hz=False, dB=False, deg=True, **mpl_kwargs):
        """Draw the Bode plot for all input-output pairs.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.
        ax
            Axis of shape (2 * `self.dim_output`, `self.dim_input`) to which to plot.
            If not given, `matplotlib.pyplot.gcf` is used to get the figure and create axis.
        Hz
            Should the frequency be in Hz on the plot.
        dB
            Should the magnitude be in dB on the plot.
        deg
            Should the phase be in degrees (otherwise in radians).
        mpl_kwargs
            Keyword arguments used in the matplotlib plot function.

        Returns
        -------
        artists
            List of matplotlib artists added.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            width, height = plt.rcParams['figure.figsize']
            fig.set_size_inches(self.dim_input * width, 2 * self.dim_output * height)
            fig.set_constrained_layout(True)
            ax = fig.subplots(2 * self.dim_output, self.dim_input, sharex=True, squeeze=False)
        else:
            assert isinstance(ax, np.ndarray) and ax.shape == (2 * self.dim_output, self.dim_input)
            fig = ax[0, 0].get_figure()

        w = np.asarray(w)
        freq = w / (2 * np.pi) if Hz else w
        freq = freq / self.sampling_time if self.sampling_time > 0 else freq
        mag, phase = self.bode(w, mu=mu)
        if deg:
            phase *= 180 / np.pi

        artists = np.empty_like(ax)
        freq_label = f'Frequency ({"Hz" if Hz else "rad/s"})'
        mag_label = f'Magnitude{" (dB)" if dB else ""}'
        phase_label = f'Phase ({"deg" if deg else "rad"})'
        for i in range(self.dim_output):
            for j in range(self.dim_input):
                if dB:
                    artists[2 * i, j] = ax[2 * i, j].semilogx(freq, 20 * np.log10(mag[:, i, j]),
                                                              **mpl_kwargs)
                else:
                    artists[2 * i, j] = ax[2 * i, j].loglog(freq, mag[:, i, j],
                                                            **mpl_kwargs)
                artists[2 * i + 1, j] = ax[2 * i + 1, j].semilogx(freq, phase[:, i, j],
                                                                  **mpl_kwargs)
        for i in range(self.dim_output):
            ax[2 * i, 0].set_ylabel(mag_label)
            ax[2 * i + 1, 0].set_ylabel(phase_label)
        for j in range(self.dim_input):
            ax[-1, j].set_xlabel(freq_label)
        fig.suptitle('Bode plot')

        return artists

    def mag_plot(self, w, mu=None, ax=None, ord=None, Hz=False, dB=False, **mpl_kwargs):
        """Draw the magnitude plot.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.
        ax
            Axis to which to plot.
            If not given, `matplotlib.pyplot.gca` is used.
        ord
            The order of the norm used to compute the magnitude (the default is the Frobenius norm).
        Hz
            Should the frequency be in Hz on the plot.
        dB
            Should the magnitude be in dB on the plot.
        mpl_kwargs
            Keyword arguments used in the matplotlib plot function.

        Returns
        -------
        out
            List of matplotlib artists added.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        w = np.asarray(w)
        freq = w / (2 * np.pi) if Hz else w
        freq = freq / self.sampling_time if self.sampling_time > 0 else freq
        mag = spla.norm(self.freq_resp(w, mu=mu), ord=ord, axis=(1, 2))
        if dB:
            out = ax.semilogx(freq, 20 * np.log10(mag), **mpl_kwargs)
        else:
            out = ax.loglog(freq, mag, **mpl_kwargs)

        ax.set_title('Magnitude plot')
        freq_unit = ' (Hz)' if Hz else ' (rad/s)'
        ax.set_xlabel('Frequency' + freq_unit)
        mag_unit = ' (dB)' if dB else ''
        ax.set_ylabel('Magnitude' + mag_unit)

        return out

    @cached
    def h2_norm(self, return_norm_only=True, **quad_kwargs):
        """Compute the H2-norm using quadrature.

        This method uses `scipy.integrate.quad` and makes no assumptions on the form of the transfer
        function.
        It only assumes that `self.tf` is defined over the imaginary axis.

        By default, the absolute error tolerance in `scipy.integrate.quad` is set to zero (see its
        optional argument `epsabs`).
        It can be changed by using the `epsabs` keyword argument.

        Parameters
        ----------
        return_norm_only
            Whether to only return the approximate H2-norm.
        quad_kwargs
            Keyword arguments passed to `scipy.integrate.quad`.

        Returns
        -------
        norm
            Computed H2-norm.
        norm_relerr
            Relative error estimate (returned if `return_norm_only` is `False`).
        info
            Quadrature info (returned if `return_norm_only` is `False` and `full_output` is `True`).
            See `scipy.integrate.quad` documentation for more details.
        """
        if self.sampling_time > 0:
            raise NotImplementedError

        import scipy.integrate as spint
        quad_kwargs.setdefault('epsabs', 0)
        quad_out = spint.quad(lambda w: spla.norm(self.eval_tf(w * 1j))**2,
                              0, np.inf,
                              **quad_kwargs)
        norm = np.sqrt(quad_out[0] / np.pi)
        if return_norm_only:
            return norm
        norm_relerr = quad_out[1] / (2 * quad_out[0])
        if len(quad_out) == 2:
            return norm, norm_relerr
        else:
            return norm, norm_relerr, quad_out[2:]

    def h2_inner(self, lti):
        """Compute H2 inner product with an |LTIModel|.

        Uses the inner product formula based on the pole-residue form
        (see, e.g., Lemma 1 in :cite:`ABG10`).
        It assumes that `self.tf` is defined on `-lti.poles()`.

        Parameters
        ----------
        lti
            |LTIModel| consisting of |Operators| that can be converted to |NumPy arrays|.
            The D operator is ignored.

        Returns
        -------
        inner
            H2 inner product.
        """
        from pymor.models.iosys import LTIModel, _lti_to_poles_b_c
        assert isinstance(lti, LTIModel)

        poles, b, c = _lti_to_poles_b_c(lti)
        inner = sum(c[i].dot(self.eval_tf(-poles[i]).dot(b[i]))
                    for i in range(len(poles)))
        inner = inner.conjugate()

        return inner

    def __add__(self, other):
        assert isinstance(other, TransferFunction) or hasattr(other, 'transfer_function')
        if not isinstance(other, TransferFunction):
            other = other.transfer_function
        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) + other.eval_tf(s, mu=mu)
        dtf = (lambda s, mu=None: self.eval_dtf(s, mu=mu) + other.eval_dtf(s, mu=mu)
               if hasattr(other, 'eval_dtf')
               else None)
        return self.with_(tf=tf, dtf=dtf)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        assert isinstance(other, TransferFunction) or hasattr(other, 'transfer_function')
        if not isinstance(other, TransferFunction):
            other = other.transfer_function
        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) - self.eval_tf(s, mu=mu)
        dtf = (lambda s, mu=None: other.eval_dtf(s, mu=mu) - self.eval_dtf(s, mu=mu)
               if hasattr(other, 'eval_dtf')
               else None)
        return self.with_(tf=tf, dtf=dtf)

    def __neg__(self):
        tf = lambda s, mu=None: -self.eval_tf(s, mu=mu)
        dtf = (lambda s, mu=None: -self.eval_dtf(s, mu=mu)) if self.dtf is not None else None
        return self.with_(tf=tf, dtf=dtf)

    def __mul__(self, other):
        assert isinstance(other, TransferFunction) or hasattr(other, 'transfer_function')
        if not isinstance(other, TransferFunction):
            other = other.transfer_function
        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_output

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) @ other.eval_tf(s, mu=mu)
        dtf = (lambda s, mu=None: (self.eval_dtf(s, mu=mu) @ other.eval_tf(s, mu=mu)
                                   + self.eval_tf(s, mu=mu) @ other.eval_dtf(s, mu=mu))
               if hasattr(other, 'eval_dtf')
               else None)
        return self.with_(tf=tf, dtf=dtf)

    def __rmul__(self, other):
        assert isinstance(other, TransferFunction) or hasattr(other, 'transfer_function')
        if not isinstance(other, TransferFunction):
            other = other.transfer_function
        assert self.sampling_time == other.sampling_time
        assert self.dim_output == other.dim_input

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) @ self.eval_tf(s, mu=mu)
        dtf = (lambda s, mu=None: (other.eval_dtf(s, mu=mu) @ self.eval_tf(s, mu=mu)
                                   + other.eval_tf(s, mu=mu) @ self.eval_dtf(s, mu=mu))
               if hasattr(other, 'eval_dtf')
               else None)
        return self.with_(tf=tf, dtf=dtf)


class FactorizedTransferFunction(TransferFunction):
    r"""Transfer functions in generalized coprime factor form.

    This class describes input-output systems given by a transfer
    function of the form
    :math:`H(s, \mu) = \mathcal{C}(s, \mu) \mathcal{K}(s, \mu)^{-1} \mathcal{B}(s, \mu)
    + \mathcal{D}(s, \mu)`.

    Parameters
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    K, B, C, D
        Functions that take `s` and return an |Operator|.
    dK, dB, dC, dD
        Functions that take `s` and return an |Operator| that is the derivative of K, B, C, D
        (optional).
    parameters
        The |Parameters| of the transfer function.
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
    name
        Name of the system.
    """

    def __init__(self, dim_input, dim_output, K, B, C, D, dK=None, dB=None, dC=None, dD=None,
                 parameters={}, sampling_time=0, name=None):
        def tf(s, mu=None):
            if dim_input <= dim_output:
                B_vec = B(s).as_range_array(mu=mu)
                Kinv_B = K(s).apply_inverse(B_vec, mu=mu)
                res = C(s).apply(Kinv_B, mu=mu).to_numpy().T
            else:
                C_vec_adj = C(s).as_source_array(mu=mu).conj()
                Kinvadj_Cadj = K(s).apply_inverse_adjoint(C_vec_adj, mu=mu)
                res = B(s).apply_adjoint(Kinvadj_Cadj, mu=mu).to_numpy().conj()
            res += to_matrix(D(s), format='dense', mu=mu)
            return res

        if dK is None or dB is None or dC is None:
            dtf = None
        else:
            def dtf(s, mu=None):
                if dim_input <= dim_output:
                    B_vec = B(s).as_range_array(mu=mu)
                    Ki_B = K(s).apply_inverse(B_vec, mu=mu)
                    dC_Ki_B = dC(s).apply(Ki_B, mu=mu).to_numpy().T

                    dB_vec = dB(s).as_range_array(mu=mu)
                    Ki_dB = K(s).apply_inverse(dB_vec, mu=mu)
                    C_Ki_dB = C(s).apply(Ki_dB, mu=mu).to_numpy().T

                    dK_Ki_B = dK(s).apply(Ki_B, mu=mu)
                    Ki_dK_Ki_B = K(s).apply_inverse(dK_Ki_B, mu=mu)
                    C_Ki_dK_Ki_B = C(s).apply(Ki_dK_Ki_B, mu=mu).to_numpy().T

                    res = dC_Ki_B + C_Ki_dB - C_Ki_dK_Ki_B
                else:
                    C_vec_a = C(s).as_source_array(mu=mu).conj()
                    Kia_Ca = K(s).apply_inverse_adjoint(C_vec_a, mu=mu)
                    dC_Ki_B = dB(s).apply_adjoint(Kia_Ca, mu=mu).to_numpy().conj()

                    dC_vec_a = dC(s).as_source_array(mu=mu).conj()
                    Kia_dCa = K(s).apply_inverse_adjoint(dC_vec_a, mu=mu)
                    CKidB = B(s).apply_adjoint(Kia_dCa, mu=mu).to_numpy().conj()

                    dKa_Kiajd_Ca = dK(s).apply_adjoint(Kia_Ca, mu=mu)
                    Kia_dKa_Kia_Ca = K(s).apply_inverse_adjoint(dKa_Kiajd_Ca, mu=mu)
                    C_Ki_dK_Ki_B = B(s).apply_adjoint(Kia_dKa_Kia_Ca, mu=mu).to_numpy().conj()

                    res = dC_Ki_B + CKidB - C_Ki_dK_Ki_B
                res += to_matrix(dD(s), format='dense', mu=mu)
                return res

        super().__init__(dim_input, dim_output, tf, dtf=dtf, parameters=parameters,
                         sampling_time=sampling_time, name=name)
        self.__auto_init(locals())

    def __add__(self, other):
        if (type(other) is not FactorizedTransferFunction
                and not hasattr(other, 'transfer_function')):
            return NotImplemented

        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        if not type(other) is FactorizedTransferFunction:
            other = other.transfer_function

        K = lambda s: BlockDiagonalOperator([self.K(s), other.K(s)])
        B = lambda s: BlockColumnOperator([self.B(s), other.B(s)])
        C = lambda s: BlockRowOperator([self.C(s), other.C(s)])
        D = lambda s: self.D(s) + other.D(s)
        dK = (lambda s: BlockDiagonalOperator([self.dK(s), other.dK(s)])
              if self.dK is not None and other.dK is not None
              else None)
        dB = (lambda s: BlockColumnOperator([self.dB(s), other.dB(s)])
              if self.dB is not None and other.dB is not None
              else None)
        dC = (lambda s: BlockRowOperator([self.dC(s), other.dC(s)])
              if self.dC is not None and other.dC is not None
              else None)
        dD = (lambda s: self.dD(s) + other.dD(s)
              if self.dD is not None and other.dD is not None
              else None)

        return self.with_(K=K, B=B, C=C, D=D, dK=dK, dB=dB, dC=dC, dD=dD)

    __radd__ = __add__

    def __neg__(self):
        C = lambda s: -self.C(s)
        D = lambda s: -self.D(s)
        dC = lambda s: -self.dC(s) if self.dC is not None else None
        dD = lambda s: -self.dD(s) if self.dD is not None else None
        return self.with_(C=C, D=D, dC=dC, dD=dD)

    def __mul__(self, other):
        if (type(other) is not FactorizedTransferFunction
                and not hasattr(other, 'transfer_function')):
            return NotImplemented

        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_output

        if not type(other) is FactorizedTransferFunction:
            other = other.transfer_function

        K = lambda s: BlockOperator([[self.K(s), -self.B(s) @ other.C(s)],
                                     [None, other.K(s)]])
        B = lambda s: BlockColumnOperator([self.B(s) @ other.D(s), other.B(s)])
        C = lambda s: BlockRowOperator([self.C(s), self.D(s) @ other.C(s)])
        D = lambda s: self.D(s) @ other.D(s)
        dK = (lambda s: BlockOperator([[self.dK(s), self.dB(s) @ other.C(s) + self.B(s) @ other.dC(s)],
                                       [None, other.dK(s)]])
              if self.dK is not None and other.dK is not None and self.dB is not None and other.dC is not None
              else None)
        dB = (lambda s: BlockColumnOperator([self.dB(s) @ other.D(s) + self.B(s) @ other.dD(s), other.dB(s)])
              if self.dB is not None and other.dB is not None and other.dD is not None
              else None)
        dC = (lambda s: BlockRowOperator([self.dC(s), self.dD(s) @ other.C(s) + self.D(s) @ other.dC(s)])
              if self.dC is not None and other.dC is not None and self.dD is not None
              else None)
        dD = (lambda s: self.dD(s) @ other.D(s) + self.D(s) @ other.dD(s)
              if self.dD is not None and other.dD is not None
              else None)

        return self.with_(K=K, B=B, C=C, D=D, dK=dK, dB=dB, dC=dC, dD=dD)

    def __rmul__(self, other):
        if not hasattr(other, 'transfer_function'):
            return NotImplemented
        return other.transfer_function * self
