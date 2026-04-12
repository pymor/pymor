# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.data_driven import DataDrivenReductor


class DDRBModelHierarchy(Model):
    def __init__(self, fom, rb_reductor, dd_reductor_parameters, tol, compression=None,
                 retrain_interval=1):
        self.__auto_init(locals())
        self.__dict__.pop('dd_model', None)

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
        """Extend DD reductors with new training data and retrain if needed.

        Training data is always accumulated in the reductor. When `retrain_interval`
        new samples have been added, the regressor is retrained from scratch.
        """
        sum_dims = 0
        for i, red in enumerate(self.dd_reductors):
            coeffs = reduced_coefficients[sum_dims:sum_dims+red.dim_solution_space]
            red.extend_training_data([mu], coeffs)
            self._pending_retrains[i] += 1
            if self._pending_retrains[i] >= self.retrain_interval:
                self.dd_models[i] = red.reduce()
                self._pending_retrains[i] = 0
            sum_dims += red.dim_solution_space

    def _solve_reduced(self, data, mu, update=True):
        """Solve the hierarchy and return the reduced solution.

        The reduced solution, used model type, and estimated error are stored in `data`.
        If the FOM fallback is used, `data['solution']` is set to the FOM solution directly.

        If `update` is `True`, bases and training data are extended as needed.
        If `update` is `False`, only the error estimation is performed without
        modifying the hierarchy.
        """
        if '_estimated_error' in data:
            return

        # Compute ML solution and estimate error
        if len(self.dd_models) == 0:
            dd_solution = self._rb_model.solution_space.zeros()
        else:
            dd_solution_np = np.vstack([dd_model.solve(mu).to_numpy() for dd_model in self.dd_models])
            dd_solution = self._rb_model.solution_space.make_array(dd_solution_np)
        dd_estimated_error = self._rb_model.error_estimator.estimate_error(dd_solution, mu, self._rb_model)
        self.logger.info(f'Estimated error of ML: {dd_estimated_error}')

        if dd_estimated_error <= self.tol:
            data['_reduced_solution'] = dd_solution
            data['_used_model'] = 'ML'
            data['_estimated_error'] = dd_estimated_error
            return

        # Compute RB solution and estimate error
        rb_solution = self._rb_model.solve(mu)
        rb_estimated_error = self._rb_model.error_estimator.estimate_error(rb_solution, mu, self._rb_model)
        self.logger.info(f'Estimated error of RB: {rb_estimated_error}')

        if rb_estimated_error <= self.tol:
            data['_reduced_solution'] = rb_solution
            data['_used_model'] = 'RB'
            data['_estimated_error'] = rb_estimated_error

            if update:
                self._update_dd_models(mu, rb_solution.to_numpy())

            return

        if not update:
            data['_estimated_error'] = rb_estimated_error
            return

        # Compute FOM solution
        fom_solution = self.fom.solve(mu)
        data['solution'] = fom_solution
        data['_used_model'] = 'FOM'
        data['_estimated_error'] = np.array([0.])

        # Extend reduced basis of RB reductor and reduce again
        old_rb_size = len(self.rb_reductor.bases['RB'])
        extension_data = fom_solution
        if self.compression is not None:  # Perform additional compression if desired
            extension_data = self.compression(fom_solution)
        self.rb_reductor.extend_basis(extension_data)
        self._rb_model = self.rb_reductor.reduce()

        # Project FOM solution onto reduced basis
        projected_fom_solution = self.rb_reductor.bases['RB'].inner(fom_solution)

        # Extend training data of existing data driven reductors
        self._update_dd_models(mu, projected_fom_solution[:old_rb_size])

        # Add new data driven reductor and model to account for new basis components
        self.dd_reductors.append(DataDrivenReductor([mu], projected_fom_solution[old_rb_size:],
                                                    **self.dd_reductor_parameters))
        self.dd_models.append(self.dd_reductors[-1].reduce())
        self._pending_retrains.append(0)

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
