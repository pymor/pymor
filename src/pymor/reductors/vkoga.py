from collections.abc import Callable
from numbers import Number

import numpy as np
import numpy.typing as npt
from vkoga.kernels import Gaussian
from vkoga.vkoga import VKOGA

from pymor.core.base import BasicObject
from pymor.models.black_box import BlackBoxModel
from pymor.parameters.base import Mu, ParameterSpace
from pymor.vectorarrays.interface import VectorSpace


class VkogaStateReductor(BasicObject):
    def __init__(
        self,
        solution_space,
        parameter_space,
        solution_length=1,
        parameter_scaling: Callable[[npt.ArrayLike], npt.ArrayLike]=lambda x: x,
        rom_name='VKOGAStateModel',
        max_iter=None,
        tol_p=1e-10,
        kernel_par=1,
        greedy_type='p_greedy',
        vkoga_verbose=False,
        training_data=None,
    ):
        assert isinstance(solution_space, VectorSpace)
        assert isinstance(parameter_space, ParameterSpace)

        # we need to
        # - rescale to [-1, 1] for kernel methods
        # - taking the parameter_scaling into account beforehand
        parameter_bounds = [[], []]
        for kk in parameter_space.ranges:
            parameter_bounds[0].append(parameter_scaling(parameter_space.ranges[kk][0]))
            parameter_bounds[1].append(parameter_scaling(parameter_space.ranges[kk][1]))
        parameter_bounds = np.array(parameter_bounds)

        self._input_scaling = (
            lambda x: 2.0
            * (parameter_scaling(x) - parameter_bounds[0, :])
            / (parameter_bounds[1, :] - parameter_bounds[0, :])
            - 1.0
        )

        self.__auto_init(locals())
        self._data = self._parse_data(training_data)

        # Even though it does not fit perfectly, we implement bases to mimic other reductors.
        self.bases = {
            # TODO: we need something with a len() here, do
            #       (mu, state) for mu, state in zip(self._data)?
            'DATA': self._data[0],
        }

        self._mlms = {}

    def extend_training_data(
        self, training_inputs, training_states
    ):
        additional_training_inputs, additional_training_states = self._parse_data(
            (training_inputs, training_states)
        )
        if len(additional_training_inputs) > 0:
            self._data[0] += additional_training_inputs
            self._data[1] += additional_training_states

    def reduce(self, dims=None):
        """Fits a kernel model using VKOGA to the amount of data specified by dims."""
        if dims is None:
            dims = {k: len(v) for k, v in self.bases.items()}
        if isinstance(dims, Number):
            dims = {k: dims for k in self.bases}
        if set(dims.keys()) != set(self.bases.keys()):
            raise ValueError(f'Must specify dimensions for {set(self.bases.keys())}')
        for k, d in dims.items():
            if d < 0:
                raise ValueError(
                    f'Reduced state dimension must be larger than zero {k}'
                )
            if d > len(self.bases[k]):
                raise ValueError(
                    f'Specified reduced state dimension larger than reduced basis {k}'
                )
        dim = dims['DATA']
        if dim not in self._mlms:
            vkoga_mlm = self._build_VKOGA()
            if dim > 0:
                training_inputs = [mu.to_numpy() for mu in self._data[0][:dim]]
                assert np.all(mu.shape == (self.parameter_space.parameters.dim,) for mu in training_inputs)
                training_inputs = np.stack(training_inputs)
                training_states = np.stack(
                    [state.ravel() for state in self._data[1][:dim]]
                )
                vkoga_mlm = self._build_VKOGA()
                with self.logger.block(
                    f'Fitting VKOGA model to {"all" if dim == len(self._data[0]) else "first"} {dim} samples ...'
                ):
                    vkoga_mlm.fit(
                        self._input_scaling(training_inputs),
                        training_states,
                        maxIter=self.max_iter or len(training_inputs),
                    )
            mlm = self._build_rom(vkoga_mlm)
            self._mlms[dim] = mlm
        return self._mlms[dim]

    def reconstruct(self, u, basis='DATA'):
        """Returns an identical copy of the MLM prediction within the solution_space."""
        assert basis in self.bases
        assert u in self.solution_space
        assert len(u) == self.solution_length
        return u.copy()

    def _build_VKOGA(self):
        vkoga_mlm = VKOGA(
            tol_p=self.tol_p,
            kernel_par=self.kernel_par,
            greedy_type=self.greedy_type,
            kernel=Gaussian(ep=1 / np.sqrt(self.parameter_space.parameters.dim)),
        )
        vkoga_mlm.verbose = not self.logging_disabled and self.vkoga_verbose
        return vkoga_mlm

    def _build_rom(self, vkoga_mlm):
        if vkoga_mlm.ctrs_ is None:
            def vkoga_trivial_solution(mu):
                return self.solution_space.zeros(self.solution_length)
            rom = BlackBoxModel(
                solution_space=self.solution_space,
                parameters=self.parameter_space.parameters,
                solve_lambda=vkoga_trivial_solution,
                name=self.rom_name,
            )
        else:
            def vkoga_predict(mu):
                raw_input = mu.to_numpy().reshape(1, -1)
                raw_state = vkoga_mlm.predict(self._input_scaling(raw_input)).reshape(
                    self.solution_space.dim, self.solution_length
                )
                u = self.solution_space.from_numpy(raw_state)
                return u
            rom = BlackBoxModel(
                solution_space=self.solution_space,
                parameters=self.parameter_space.parameters,
                solve_lambda=vkoga_predict,
                name=self.rom_name,
            )
        rom.disable_logging()
        return rom

    def _parse_data(self, data):
        if data is None:
            return [[], []]
        else:
            assert len(data) == 2
            inputs, states = data
            assert len(inputs) == len(states)
            inputs = [
                self.parameter_space.parameters.parse(mu) if not isinstance(mu, Mu) else mu
                for mu in inputs
            ]
            assert np.all(state in self.solution_space for state in states)
            assert np.all(len(state) == self.solution_length for state in states)
            states = [state.to_numpy().ravel() for state in states]
            new_inputs = []
            new_states = []
            for mu, state in zip(inputs, states):
                try:
                    already_colected_mu_index = self._data[0].index(mu)
                    already_collected_u = self._data[1][already_colected_mu_index]
                    assert np.linalg.norm(already_collected_u - state, np.inf) < 1e-14
                except ValueError:
                    # index() raises for unknown mus, so we did not collect data for this one yet
                    new_inputs.append(mu)
                    new_states.append(state)

            return [new_inputs, new_states]
