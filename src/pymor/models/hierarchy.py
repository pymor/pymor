# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.data_driven import DataDrivenReductor


class DDRBModelHierarchy(Model):
    """Adaptive model hierarchy combining data-driven and reduced basis models.

    The adaptive model hierarchy consists of three models: A full-order model,
    a reduced basis ROM and a data-driven model. For a new parameter, the hierarchy
    evaluates the data-driven model first as it is (usually) the fastest model.
    The estimated error of the data-driven model is compared to the prescribed
    tolerance. If the error is below the tolerance, the data-driven solution is
    returned without calling the other two models. If the error is larger than the
    tolerance, the hierarchy falls back to the reduced basis model that is more
    accurate compared to the data-driven model but also more costly to evaluate.
    If the error of the reduced basis model is below the tolerance, its solution
    is returned and used to improve the data-driven model. If the error is too
    large, the full-order model is called and its solution used to extend the
    reduced basis of the reduced basis model.

    The implementation is based on the strategies described in :cite:`HKOSW23`.

    Parameters
    ----------
    fom
        Full-order model used as last fallback option in the hierarchy.
    rb_reductor
        Reductor used to generate the reduced basis model.
    dd_reductor_parameters
        Attributes used to generate the
        :class:`~pymor.reductors.data_driven.DataDrivenReductor`.
    tol
        Tolerance to which the estimated errors are compared to decide which
        solution to return.
    compression
        Either `None` or a callable used to compress solutions before extending
        the reduced basis. Particularly useful when dealing with instationary
        problems where the solution trajectory should first be compressed using
        for instance the :func:`~pymor.algorithms.pod.pod` algorithm.
    retrain_interval
        Integer used to determine how many new training data points should be
        collected before training the data-driven model again.
    """

    def __init__(self, fom, rb_reductor, dd_reductor_parameters, tol, compression=None, retrain_interval=1):
        self.__auto_init(locals())

        assert isinstance(rb_reductor, ProjectionBasedReductor)
        assert compression is None or callable(compression)
        assert isinstance(retrain_interval, int)
        assert retrain_interval >= 1

        self.dd_reductors = []
        self.dd_models = []
        self._rb_model = self.rb_reductor.reduce()
        self._pending_retrains = []

        self.solution_space = fom.solution_space
        self.dim_output = fom.dim_output

        super().__init__(products=fom.products, visualizer=fom.visualizer)

    def _update_dd_models(self, mu, reduced_coefficients):
        """Extend data-driven reductors with new training data and retrain if needed.

        Training data is always accumulated in the reductor. When `retrain_interval`
        new samples have been added, the regressor is retrained from scratch.

        Parameters
        ----------
        mu
            |Parameter value| for which reduced coefficients have been computed.
        reduced_coefficients
            Reduced solution associated to `mu` that is obtained from the reduced basis model.
        """
        sum_dims = 0
        for i, red in enumerate(self.dd_reductors):
            coeffs = reduced_coefficients[sum_dims:sum_dims+red.dim_solution_space]
            red.extend_training_data([mu], coeffs.T)
            self._pending_retrains[i] += 1
            if self._pending_retrains[i] >= self.retrain_interval:
                self.dd_models[i] = red.reduce()
                self._pending_retrains[i] = 0
            sum_dims += red.dim_solution_space

    def _solve_reduced(self, data, mu, update=True):
        """Query the hierarchy and add solution to data dictionary.

        The reduced solution, used model type, and estimated error are stored in `data`.
        If the FOM fallback is used, `data['solution']` is set to the FOM solution directly.

        Parameters
        ----------
        data
            Dictionary to store the results in.
        mu
            |Parameter value| to query the hierarchy for.
        update
            If `True`, bases and training data are extended as needed and surrogates are trained.
            If `False`, only the error estimation is performed without modifying the models.
        """
        if '_estimated_error' in data:
            return

        # compute ML solution and estimate error
        if len(self.dd_models) == 0:
            nt = getattr(getattr(self._rb_model, 'time_stepper', None), 'nt', 0)
            dd_solution = self._rb_model.solution_space.zeros(nt + 1)
        else:
            dd_solution_np = np.vstack([dd_model.solve(mu).to_numpy() for dd_model in self.dd_models])
            dd_solution = self._rb_model.solution_space.make_array(dd_solution_np)
        dd_estimated_error = self._rb_model.error_estimator.estimate_error(dd_solution, mu, self._rb_model)

        if np.max(dd_estimated_error) <= self.tol:
            self.logger.info(f'Returning ML solution, estimated error of ML: {np.max(dd_estimated_error)}')
            data['_reduced_solution'] = dd_solution
            data['_used_model'] = 'ML'
            data['_estimated_error'] = dd_estimated_error
            return

        self.logger.info(f'Falling back to RB, estimated error of ML: {np.max(dd_estimated_error)}')

        # compute RB solution and estimate error
        rb_solution = self._rb_model.solve(mu)
        rb_estimated_error = self._rb_model.error_estimator.estimate_error(rb_solution, mu, self._rb_model)

        if np.max(rb_estimated_error) <= self.tol:
            self.logger.info(f'Returning RB solution, estimated error of RB: {np.max(rb_estimated_error)}')
            data['_reduced_solution'] = rb_solution
            data['_used_model'] = 'RB'
            data['_estimated_error'] = rb_estimated_error

            if update:
                with self.logger.block('Updating data-driven models ...'):
                    self._update_dd_models(mu, rb_solution.to_numpy())

            return

        data['_estimated_error'] = np.array([0.])
        if not update:
            return

        # compute FOM solution
        self.logger.info('Falling back to FOM')
        fom_solution = self.fom.solve(mu)
        data['solution'] = fom_solution
        data['_used_model'] = 'FOM'

        # extend reduced basis of RB reductor and reduce again
        old_rb_size = self._rb_model.order
        extension_data = fom_solution
        if self.compression is not None:  # Perform additional compression if desired
            extension_data = self.compression(fom_solution)
        self.rb_reductor.extend_basis(extension_data)
        self._rb_model = self.rb_reductor.reduce()

        # project FOM solution onto reduced basis
        projected_fom_solution = self.rb_reductor.bases['RB'].inner(fom_solution,
                                                                    product=self.rb_reductor.products.get('RB'))

        # extend training data of existing data driven reductors
        with self.logger.block('Updating data-driven models ...'):
            self._update_dd_models(mu, projected_fom_solution[:old_rb_size])

        # add new data driven reductor and model to account for new basis components
        T = getattr(self.fom, 'T', None)
        self.dd_reductors.append(DataDrivenReductor([mu], projected_fom_solution[old_rb_size:].T,
                                                    T=T, **self.dd_reductor_parameters))
        self.dd_models.append(self.dd_reductors[-1].reduce())
        self._pending_retrains.append(0)

        assert sum(red.dim_solution_space for red in self.dd_reductors) == self._rb_model.order

    def _compute(self, quantities, data, mu):
        update = bool(quantities & {'solution', 'output'})

        if quantities & {'solution', 'output', 'solution_error_estimate', 'output_error_estimate'}:
            self._solve_reduced(data, mu, update=update)

            if 'solution' in quantities:
                if 'solution' not in data:
                    # ML or RB path: reconstruct from reduced solution
                    data['solution'] = self.rb_reductor.reconstruct(data['_reduced_solution'])
                quantities.remove('solution')

            if 'output' in quantities:
                if data['_used_model'] == 'FOM':
                    data['output'] = self.fom.output_functional.apply(data['solution'], mu=mu).to_numpy()
                else:
                    data['output'] = self._rb_model.output_functional.apply(data['_reduced_solution'], mu=mu).to_numpy()
                quantities.remove('output')

            if 'solution_error_estimate' in quantities:
                data['solution_error_estimate'] = data['_estimated_error']
                quantities.remove('solution_error_estimate')

            if 'output_error_estimate' in quantities:
                if '_reduced_solution' in data:
                    data['output_error_estimate'] = self._rb_model.error_estimator.estimate_output_error(
                        data['_reduced_solution'], mu, self._rb_model)
                else:
                    # FOM path: no estimation error
                    data['output_error_estimate'] = np.zeros((self.dim_output, 1))
                quantities.remove('output_error_estimate')

        super()._compute(quantities, data, mu=mu)
