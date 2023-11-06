# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import chain

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import CacheableObject, cached
from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockOperator, BlockRowOperator
from pymor.parameters.base import Mu, Parameters, ParametricObject
from pymor.tools.plot import adaptive


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
    presets
        A `dict` of preset attributes or `None`. The dict must only contain keys that correspond to
        attributes of |TransferFunction| such as `h2_norm`.
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

    def __init__(self, dim_input, dim_output, tf, dtf=None, parameters={}, sampling_time=0, presets=None, name=None):
        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        self.parameters_own = parameters

        assert presets is None or presets.keys() <= {'h2_norm'}
        if presets:
            assert parameters == {}
        else:
            presets = {}

        self.__auto_init(locals())

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of inputs:  {self.dim_input}\n'
            f'    number of outputs: {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}'
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

    @cached
    def freq_resp(self, w, mu=None, adaptive_type='bode', adaptive_opts=None):
        """Evaluate the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            If `len(w) == 2`, the left and right limits used for the adaptive sampling.
            Otherwise, a sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.
        adaptive_type
            The plot type that adaptive sampling should be used for
            (`'bode'`, `'mag'`).
            Ignored if `len(w) != 2`.
        adaptive_opts
            Optional arguments for :func:`~pymor.tools.plot.adaptive` (ignored if `len(w) != 2`).
            If `xscale` and `yscale` are not set, `'log'` is used.

        Returns
        -------
        w
            A sequence of angular frequencies at which the transfer function was computed
            (returned if `len(w) == 2`).
        tfw
            Transfer function values at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        if self.sampling_time > 0 and not all(-np.pi <= wi <= np.pi for wi in w):
            self.logger.warning('Some frequencies are not in the [-pi, pi] interval.')

        if len(w) == 2:
            if adaptive_type == 'bode':
                if self.sampling_time == 0:
                    f = lambda w: self.eval_tf(1j * w, mu=mu)
                else:
                    f = lambda w: self.eval_tf(np.exp(1j * w), mu=mu)
            else:
                if self.sampling_time == 0:
                    f = lambda w: spla.norm(self.eval_tf(1j * w, mu=mu))
                else:
                    f = lambda w: spla.norm(self.eval_tf(np.exp(1j * w), mu=mu))

            if adaptive_opts is None:
                adaptive_opts = {}
            else:
                adaptive_opts = adaptive_opts.copy()
            adaptive_opts.setdefault('xscale', 'log')
            adaptive_opts.setdefault('yscale', 'log')
            w_new, _ = adaptive(f, w[0], w[1], **adaptive_opts)
        else:
            w_new = w

        w_complex = 1j * w_new if self.sampling_time == 0 else np.exp(1j * w_new)
        tfw = np.stack([self.eval_tf(wi, mu=mu) for wi in w_complex])

        if len(w) == 2:
            return w_new, tfw
        return tfw

    def bode(self, w, mu=None, adaptive_opts=None):
        """Compute magnitudes and phases.

        Parameters
        ----------
        w
            If `len(w) == 2`, the left and right limits used for the adaptive sampling.
            Otherwise, a sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.
        adaptive_opts
            Optional arguments for :func:`~pymor.tools.plot.adaptive` (ignored if `len(w) != 2`).

        Returns
        -------
        w
            A sequence of angular frequencies at which the transfer function was computed
            (returned if `len(w) == 2`).
        mag
            Transfer function magnitudes at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        phase
            Transfer function phases (in radians) at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        """
        tfw = self.freq_resp(w, mu=mu, adaptive_type='bode', adaptive_opts=adaptive_opts)
        if len(w) == 2:
            w_new, tfw = tfw
        mag = np.abs(tfw)
        phase = np.angle(tfw)
        phase = np.unwrap(phase, axis=0)
        if len(w) != 2:
            return mag, phase
        return w_new, mag, phase

    def bode_plot(self, w, mu=None, ax=None, Hz=False, dB=False, deg=True, adaptive_opts=None, input_indices=None,
                  output_indices=None, **mpl_kwargs):
        """Draw the Bode plot for all input-output pairs.

        Parameters
        ----------
        w
            If `len(w) == 2`, the left and right limits used for the adaptive sampling.
            Otherwise, a sequence of angular frequencies at which to compute the transfer function.
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
        adaptive_opts
            Optional arguments for :func:`~pymor.tools.plot.adaptive` (ignored if `len(w) != 2`).
        input_indices
            Optional argument to select specific inputs to be paired with all outputs
            or selected ones. If `None`, all inputs are used for plotting, otherwise, an
            `iterable` containing the indices of the selected inputs has to be passed. The
            order of the plots depends on the order of the indices in the `iterable`. It is
            possible to pass negative indices to access the inputs counting backwards.
        output_indices
            Optional argument to select specific outputs to be paired  with all inputs
            or selected ones. If `None`, all outputs are used for plotting, otherwise, an
            `iterable` containing the indices of the selected outputs has to be passed. The
            order of the plots depends on the order of the indices in the `iterable`. It is
            possible to pass negative indices to access the outputs counting backwards.
        mpl_kwargs
            Keyword arguments used in the matplotlib plot function.

        Returns
        -------
        artists
            List of matplotlib artists added.
        """
        if input_indices is None:
            input_indices = list(range(self.dim_input))
        if output_indices is None:
            output_indices = list(range(self.dim_output))

        assert all(isinstance(item, int) for item in chain(input_indices, output_indices))
        assert all(-self.dim_input <= item < self.dim_input for item in input_indices), \
            f'input_indices should be any integer value between {-self.dim_input} and {self.dim_input-1}'
        assert all(-self.dim_output <= item < self.dim_output for item in output_indices), \
            f'output_indices should be any integer value between {-self.dim_output} and {self.dim_output-1}'
        num_input, num_output = len(input_indices), len(output_indices)

        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            width, height = plt.rcParams['figure.figsize']
            fig.set_size_inches(num_input * width, 2 * num_output * height)
            fig.set_constrained_layout(True)
            ax = fig.subplots(2 * num_output, num_input, sharex=True, squeeze=False)
        else:
            if num_input == 1:
                ax = ax.reshape((2 * num_output, 1))
            assert isinstance(ax, np.ndarray)
            assert ax.shape == (2 * num_output, num_input), \
                f'ax.shape={ax.shape} should be ({2 * num_output}, {num_input})'
            fig = ax[0, 0].get_figure()

        if len(w) != 2:
            mag, phase = self.bode(w, mu=mu)
        else:
            w, mag, phase = self.bode(w, mu=mu, adaptive_opts=adaptive_opts)
        w = np.asarray(w)
        freq = w / (2 * np.pi) if Hz else w
        freq = freq / self.sampling_time if self.sampling_time > 0 else freq
        if deg:
            phase *= 180 / np.pi
        artists = np.empty_like(ax)
        freq_label = f'Frequency ({"Hz" if Hz else "rad/s"})'
        mag_label = f'Magnitude{" (dB)" if dB else ""}'
        phase_label = f'Phase ({"deg" if deg else "rad"})'
        for i in range(num_output):
            for j in range(num_input):
                if dB:
                    artists[2 * i, j] = ax[2 * i, j].semilogx(freq, 20 * np.log10(mag[:, output_indices[i],
                                                              input_indices[j]]), **mpl_kwargs)
                else:
                    artists[2 * i, j] = ax[2 * i, j].loglog(freq, mag[:, output_indices[i],
                                                            input_indices[j]], **mpl_kwargs)
                artists[2 * i + 1, j] = ax[2 * i + 1, j].semilogx(freq, phase[:, output_indices[i],
                                                                  input_indices[j]], **mpl_kwargs)
        for i in range(num_output):
            ax[2 * i, 0].set_ylabel(mag_label)
            ax[2 * i + 1, 0].set_ylabel(phase_label)
        for j in range(num_input):
            ax[-1, j].set_xlabel(freq_label)
        fig.suptitle('Bode plot')
        return artists

    def mag_plot(self, w, mu=None, ax=None, ord=None, Hz=False, dB=False, adaptive_opts=None, **mpl_kwargs):
        """Draw the magnitude plot.

        Parameters
        ----------
        w
            If `len(w) == 2`, the left and right limits used for the adaptive sampling.
            Otherwise, a sequence of angular frequencies at which to compute the transfer function.
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
        adaptive_opts
            Optional arguments for :func:`~pymor.tools.plot.adaptive` (ignored if `len(w) != 2`).
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

        if len(w) != 2:
            w = np.asarray(w)
            tfw = self.freq_resp(w, mu=mu)
        else:
            w, tfw = self.freq_resp(w, mu=mu, adaptive_type='mag', adaptive_opts=adaptive_opts)
        mag = spla.norm(tfw, ord=ord, axis=(1, 2))

        freq = w / (2 * np.pi) if Hz else w
        freq = freq / self.sampling_time if self.sampling_time > 0 else freq
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
    def h2_norm(self, return_norm_only=True, mu=None, **quad_kwargs):
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
        mu
            |Parameter values|.
        quad_kwargs
            Keyword arguments passed to `scipy.integrate.quad`.

        Returns
        -------
        norm
            Computed H2-norm.
        norm_relerr
            Relative error estimate (returned if `return_norm_only` is `False` and `presets` does
            not contain `'h2_norm'`).
        info
            Quadrature info (returned if `return_norm_only` is `False` and `full_output` is `True`
            and `presets` does not contain `'h2_norm'`).
            See `scipy.integrate.quad` documentation for more details.
        """
        if self.sampling_time > 0:
            raise NotImplementedError

        if 'h2_norm' in self.presets:
            return self.presets['h2_norm']

        import scipy.integrate as spint
        quad_kwargs.setdefault('epsabs', 0)
        quad_out = spint.quad(lambda w: spla.norm(self.eval_tf(w * 1j, mu=mu))**2,
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

    def h2_inner(self, lti, mu=None):
        """Compute H2 inner product with an |LTIModel|.

        Uses the inner product formula based on the pole-residue form
        (see, e.g., Lemma 1 in :cite:`ABG10`).
        It assumes that `self.tf` is defined on `-lti.poles()`.

        Parameters
        ----------
        lti
            |LTIModel| consisting of |Operators| that can be converted to |NumPy arrays|.
            The D operator is ignored.
        mu
            |Parameter values|.

        Returns
        -------
        inner
            H2 inner product.
        """
        from pymor.models.iosys import LTIModel, _lti_to_poles_b_c
        assert isinstance(lti, LTIModel)

        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        poles, b, c = _lti_to_poles_b_c(lti, mu=mu)
        inner = sum(c[i].dot(self.eval_tf(-poles[i], mu=mu).dot(b[i]))
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
        return self.with_(tf=tf, dtf=dtf, parameters=Parameters.of(self, other))

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
        return self.with_(tf=tf, dtf=dtf, parameters=Parameters.of(self, other))

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
        return self.with_(tf=tf, dtf=dtf, parameters=Parameters.of(self, other))

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
        return self.with_(tf=tf, dtf=dtf, parameters=Parameters.of(self, other))


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
            mu = Mu({'s': [s]}) if mu is None else mu.with_(s=s)
            if dim_input <= dim_output:
                B_vec = B.as_range_array(mu=mu)
                Kinv_B = K.apply_inverse(B_vec, mu=mu)
                res = C.apply(Kinv_B, mu=mu).to_numpy().T
            else:
                C_vec_adj = C.as_source_array(mu=mu).conj()
                Kinvadj_Cadj = K.apply_inverse_adjoint(C_vec_adj, mu=mu)
                res = B.apply_adjoint(Kinvadj_Cadj, mu=mu).to_numpy().conj()
            res += to_matrix(D, format='dense', mu=mu)
            return res

        if dK is None or dB is None or dC is None:
            dtf = None
        else:
            def dtf(s, mu=None):
                mu = Mu({'s': [s]}) if mu is None else mu.with_(s=s)
                if dim_input <= dim_output:
                    B_vec = B.as_range_array(mu=mu)
                    Ki_B = K.apply_inverse(B_vec, mu=mu)
                    dC_Ki_B = dC.apply(Ki_B, mu=mu).to_numpy().T

                    dB_vec = dB.as_range_array(mu=mu)
                    Ki_dB = K.apply_inverse(dB_vec, mu=mu)
                    C_Ki_dB = C.apply(Ki_dB, mu=mu).to_numpy().T

                    dK_Ki_B = dK.apply(Ki_B, mu=mu)
                    Ki_dK_Ki_B = K.apply_inverse(dK_Ki_B, mu=mu)
                    C_Ki_dK_Ki_B = C.apply(Ki_dK_Ki_B, mu=mu).to_numpy().T

                    res = dC_Ki_B + C_Ki_dB - C_Ki_dK_Ki_B
                else:
                    C_vec_a = C.as_source_array(mu=mu).conj()
                    Kia_Ca = K.apply_inverse_adjoint(C_vec_a, mu=mu)
                    dC_Ki_B = dB.apply_adjoint(Kia_Ca, mu=mu).to_numpy().conj()

                    dC_vec_a = dC.as_source_array(mu=mu).conj()
                    Kia_dCa = K.apply_inverse_adjoint(dC_vec_a, mu=mu)
                    CKidB = B.apply_adjoint(Kia_dCa, mu=mu).to_numpy().conj()

                    dKa_Kiajd_Ca = dK.apply_adjoint(Kia_Ca, mu=mu)
                    Kia_dKa_Kia_Ca = K.apply_inverse_adjoint(dKa_Kiajd_Ca, mu=mu)
                    C_Ki_dK_Ki_B = B.apply_adjoint(Kia_dKa_Kia_Ca, mu=mu).to_numpy().conj()

                    res = dC_Ki_B + CKidB - C_Ki_dK_Ki_B
                res += to_matrix(dD, format='dense', mu=mu)
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

        if type(other) is not FactorizedTransferFunction:
            other = other.transfer_function

        K = BlockDiagonalOperator([self.K, other.K])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, other.C])
        D = self.D + other.D
        dK = (BlockDiagonalOperator([self.dK, other.dK])
              if self.dK is not None and other.dK is not None
              else None)
        dB = (BlockColumnOperator([self.dB, other.dB])
              if self.dB is not None and other.dB is not None
              else None)
        dC = (BlockRowOperator([self.dC, other.dC])
              if self.dC is not None and other.dC is not None
              else None)
        dD = (self.dD + other.dD
              if self.dD is not None and other.dD is not None
              else None)

        return self.with_(K=K, B=B, C=C, D=D, dK=dK, dB=dB, dC=dC, dD=dD, parameters=Parameters.of(self, other))

    __radd__ = __add__

    def __neg__(self):
        C = -self.C
        D = -self.D
        dC = -self.dC if self.dC is not None else None
        dD = -self.dD if self.dD is not None else None
        return self.with_(C=C, D=D, dC=dC, dD=dD)

    def __mul__(self, other):
        if (type(other) is not FactorizedTransferFunction
                and not hasattr(other, 'transfer_function')):
            return NotImplemented

        assert self.sampling_time == other.sampling_time
        assert self.dim_input == other.dim_output

        if type(other) is not FactorizedTransferFunction:
            other = other.transfer_function

        K = BlockOperator([[self.K, -self.B @ other.C],
                                     [None, other.K]])
        B = BlockColumnOperator([self.B @ other.D, other.B])
        C = BlockRowOperator([self.C, self.D @ other.C])
        D = self.D @ other.D
        dK = (BlockOperator([[self.dK, self.dB @ other.C + self.B @ other.dC],
                                       [None, other.dK]])
              if self.dK is not None and other.dK is not None and self.dB is not None and other.dC is not None
              else None)
        dB = (BlockColumnOperator([self.dB @ other.D + self.B @ other.dD, other.dB])
              if self.dB is not None and other.dB is not None and other.dD is not None
              else None)
        dC = (BlockRowOperator([self.dC, self.dD @ other.C + self.D @ other.dC])
              if self.dC is not None and other.dC is not None and self.dD is not None
              else None)
        dD = (self.dD @ other.D + self.D @ other.dD
              if self.dD is not None and other.dD is not None
              else None)

        return self.with_(K=K, B=B, C=C, D=D, dK=dK, dB=dB, dC=dC, dD=dD, parameters=Parameters.of(self, other))

    def __rmul__(self, other):
        if not hasattr(other, 'transfer_function'):
            return NotImplemented
        return other.transfer_function * self
