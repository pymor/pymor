# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model


class ModelHierarchy(Model):
    r"""Adaptive hierarchy of models of increasing fidelity.

    The hierarchy consists of a sequence of reduced models of increasing fidelity
    together with a high-fidelity reference model (typically the full-order model)
    that serves as the final fallback. For a new parameter, the cheapest model is
    evaluated first and the accuracy of its result is verified by means of an a
    posteriori error estimator. If the estimated error is smaller than the prescribed
    tolerance `tol`, the corresponding (reconstructed) solution is returned. Otherwise,
    the hierarchy falls back to the next more accurate model, until either a model is
    accurate enough or the reference model is reached.

    The hierarchy always tries the faster models first, until a solution is obtained
    that fulfills the accuracy requirement. Whenever the hierarchy has to fall back to
    a more accurate model, the cheaper models below it are adapted using the more
    accurate solution as training data (e.g. the reduced basis is extended or a
    data-driven surrogate is retrained). The hierarchy therefore starts with empty
    reduced models and improves them on demand. Models are only adapted when an actual
    solution or output is requested; queries for error estimates alone leave the
    hierarchy unchanged.

    This hierarchy is not restricted to a fixed set of models: any number of adaptive
    reductors can be combined, e.g., only a reduced basis reductor together with a
    full-order model, or a data-driven surrogate on top of a reduced basis model. The
    implementation is based on the strategies described in :cite:`HKOSW23`.

    Parameters
    ----------
    fom
        The high-fidelity reference model used as the final fallback.
    reductor_factories
        Sequence of callables, ordered from the highest-fidelity reduced model down to
        the cheapest one (i.e. the reductor right below the reference model first). Each
        callable is passed the next higher-fidelity model (starting with `fom`) and has
        to return an adaptive reductor for the level below it. A reductor has to provide
        `reduce`, `reconstruct` and `adapt` methods (see
        :class:`~pymor.reductors.basic.ProjectionBasedReductor` and
        :class:`~pymor.reductors.data_driven.AdaptiveDataDrivenReductor`).
    tol
        Tolerance against which the estimated errors are compared to decide which
        model's solution to return.
    time_reduction
        Callable mapping a (possibly time-dependent) error estimate to a single scalar
        that is compared against `tol`. For instationary problems the error estimate is a
        trajectory over time; the default `numpy.max` uses the maximum in time (i.e. the
        :math:`\ell^\infty`-in-time norm). Other choices (e.g. the value at the final time
        or an :math:`\ell^2`-in-time norm) can be passed here.
    """

    def __init__(self, fom, reductor_factories, tol, time_reduction=np.max):
        assert len(reductor_factories) >= 1
        assert all(callable(rf) for rf in reductor_factories)
        assert tol > 0
        assert callable(time_reduction)

        models = [fom]
        reductors = []
        for rf in reductor_factories:
            reductor = rf(models[-1])
            reductors.append(reductor)
            models.append(reductor.reduce())

        models = list(reversed(models))
        reductors = list(reversed(reductors))

        assert all(m.error_estimator is not None for m in models[:-1]), 'all surrogates must provide an error estimator'

        reference_model = models[-1]
        super().__init__(dim_input=reference_model.dim_input, products=reference_model.products,
                         visualizer=reference_model.visualizer)
        self.solution_space = reference_model.solution_space
        self.dim_output = reference_model.dim_output
        self.models = models
        self.reductors = reductors
        self.__auto_init(locals())

    def _select_model_and_compute(self, mu, quantities):
        base_quantities = quantities & {'solution', 'output'}

        errors_to_compute = set()
        if quantities & {'solution', 'solution_error_estimate'}:
            errors_to_compute.add('solution_error_estimate')
        if quantities & {'output', 'output_error_estimate'}:
            errors_to_compute.add('output_error_estimate')

        def accurate_enough(result):
            # checks if all requested error estimates are available and below the tolerance
            return all((est := result.get(q)) is not None and self.time_reduction(est) <= self.tol
                       for q in errors_to_compute)

        # iterate over the models to determine a sufficiently accurate solution and/or output
        for i_m, m in enumerate(self.models):
            # check if the current model is the reference model
            is_reference = i_m == len(self.models) - 1

            # determine the requested quantities and call the `compute`-method of the current model
            requested = base_quantities if is_reference else base_quantities | errors_to_compute
            result = m.compute(**dict.fromkeys(requested, True), mu=mu)

            # fill missing error estimates when the reference model was reached
            if is_reference:
                self._fill_zero_error_estimates(result, quantities)

            # accept the result by leaving the loop if the reference model was reached
            # or the estimated errors in solution and/or output are below the tolerance
            if is_reference or accurate_enough(result):
                break

        i_m_sufficient = i_m

        return result, i_m_sufficient

    def _fill_zero_error_estimates(self, data, quantities):
        error_estimates = {'solution_error_estimate', 'output_error_estimate'}
        # the reference model is exact: report a zero error, shaped like the corresponding
        # estimate, for any requested estimate that no model produced
        for quantity in quantities & error_estimates:
            if quantity == 'solution_error_estimate' and 'solution' in data:
                data[quantity] = np.zeros(len(data['solution']))
            elif quantity == 'output_error_estimate' and 'output' in data:
                data[quantity] = np.zeros_like(data['output'])
            else:
                data[quantity] = np.zeros(1)

    def _reconstruct(self, data, i_m_sufficient, quantities, result):
        data['used_model'] = len(self.models) - 1 - i_m_sufficient
        # if the solution is requested, reconstruct it such that it lives
        # in the solution space of the reference model
        if 'solution' in quantities:
            solution = result['solution']
            # iteratively reconstruct through all reductors in the hierarchy starting from the one
            # whose model was the first to produce a sufficiently accurate solution
            for red in self.reductors[i_m_sufficient:]:
                solution = red.reconstruct(solution)  # could be a nop
            data['solution'] = solution

    def _adapt_lower_fidelity_models(self, mu, i_m_sufficient, quantities):
        # iteratively adapt the lower fidelity models using the higher fidelity data
        # from the previous model in the hierarchy
        m_new = None
        for i in range(i_m_sufficient-1, -1, -1):
            adapt_data = self.models[i+1].compute(**dict.fromkeys(quantities, True), mu=mu)
            # call the `adapt`-method of the respective reductor
            m_new = self.reductors[i].adapt(mu, new_fom=m_new,
                                            fom_solution=adapt_data.get('solution'),
                                            fom_output=adapt_data.get('output'))
            # end the loop if the model did not change
            if self.models[i] == m_new:
                return
            # update the model in the models list
            self.models[i] = m_new

    def _compute(self, quantities, data, mu):
        result, i_m_sufficient = self._select_model_and_compute(mu, quantities)
        data.update(result)

        self._reconstruct(data, i_m_sufficient, quantities, result)

        base_quantities = quantities & {'solution', 'output'}
        if base_quantities and i_m_sufficient != 0:
            self._adapt_lower_fidelity_models(mu, i_m_sufficient, base_quantities)

        quantities -= data.keys()
        super()._compute(quantities, data, mu=mu)
