from typing import Optional, Union, Iterable

import sklearn.base as skbase
import sklearn.utils as skutils

from collections.abc import Callable
from numbers import Number

import numpy as np
from vkoga.kernels import Gaussian, Matern
from vkoga.vkoga import VKOGA

from pymor.core.base import BasicObject
from pymor.models.generic import GenericModel
from pymor.parameters.base import Mu, Parameters
from pymor.tools.random import get_rng
from pymor.vectorarrays.interface import VectorSpace, VectorArray


class VKOGAEstimator(VKOGA, BasicObject):
    def __init__(
        self,
        kernel_type="gaussian",
        kernel_length_scale=1.0,
        greedy_type="p_greedy",
        greedy_tol=1e-10,
        greedy_max_iter=100,
        regularisation_parameter=0.0,
        selection_threshold=0.0,
        verbose=False,
    ):
        if kernel_type == "gaussian":
            kernel = Gaussian(ep=kernel_length_scale)
        elif kernel_type[:6] == "matern":
            order = int(kernel_type[7:]) if len(kernel_type) > 6 else 1
            kernel = Matern(ep=kernel_length_scale, k=order)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")
        super().__init__(
            kernel=kernel,
            kernel_par=1,  # unsused?
            verbose=verbose,
            n_report=10,
            greedy_type=greedy_type,
            reg_par=regularisation_parameter,
            restr_par=selection_threshold,
            tol_f=greedy_tol,
            tol_p=greedy_tol,
        )
        self.__auto_init(locals())

    def fit(self, X, y):
        # we only need the amout of data, not the data itself, but it seems easiest to let sklearn to the parsing
        X, y = skutils.validation.check_X_y(X, y, multi_output=True)
        N, _ = y.shape
        max_iter = min(self.greedy_max_iter, N)
        super().fit(X, y, maxIter=max_iter)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "ctrs_") and self.ctrs_ is not None

    def set_params(self, **params):
        """Set parameters - required for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)

        # Reinitialize kernel when parameters change
        if "kernel_type" in params or "kernel_length_scale" in params:
            if self.kernel_type == "gaussian":
                self.kernel = Gaussian(ep=self.kernel_length_scale)
            elif self.kernel_type[:6] == "matern":
                order = int(self.kernel_type[7:]) if len(self.kernel_type) > 6 else 1
                self.kernel = Matern(ep=self.kernel_length_scale, k=order)

        # Update VKOGA parameters
        self.tol_f = self.greedy_tol
        self.tol_p = self.greedy_tol
        self.reg_par = self.regularisation_parameter
        self.restr_par = self.selection_threshold

        return self

    def get_params(self, deep=True):
        """Get parameters - required for sklearn compatibility"""
        return {
            "kernel_type": self.kernel_type,
            "kernel_length_scale": self.kernel_length_scale,
            "greedy_type": self.greedy_type,
            "greedy_tol": self.greedy_tol,
            "greedy_max_iter": self.greedy_max_iter,
            "regularisation_parameter": self.regularisation_parameter,
            "selection_threshold": self.selection_threshold,
            "verbose": self.verbose,
        }


class DataDrivenModel(GenericModel):
    def __init__(
        self,
        parameters,
        parameter_scaling,
        compute_shapes,
        estimators,
        name: Optional[str] = "DataDrivenModel",
    ):
        self.__auto_init(locals())
        computers = {}
        for compute_id, estimator in self.estimators.items():
            data_dim, data_length = self.compute_shapes[compute_id]
            if not skutils.validation._is_fitted(estimator):
                if isinstance(data_dim, VectorSpace):

                    def trivial_solution(mu):
                        return data_dim.zeros(data_length)
                else:

                    def trivial_solution(mu):
                        return np.zeros((data_dim, data_length))

                computers[compute_id] = (data_dim, trivial_solution)
            else:
                if isinstance(data_dim, VectorSpace):

                    def predict(mu):
                        mu = self.parameter_scaling(mu)
                        X = mu.to_numpy().reshape(1, -1)
                        Y = estimator.predict(X)
                        return data_dim.from_numpy(Y.reshape(data_length, data_dim.dim))
                else:

                    def predict(mu):
                        mu = self.parameter_scaling(mu)
                        X = mu.to_numpy().reshape(1, -1)
                        Y = estimator.predict(X)
                        return Y.reshape((data_length, data_dim))

                computers[compute_id] = (data_dim, predict)
        super().__init__(parameters=parameters, computers=computers, name=name)


class DataDrivenReductor(BasicObject):
    def __init__(
        self,
        *,
        parameters: Parameters,
        compute_shapes: dict[
            str, tuple[VectorSpace | Number, Number]
        ],  # compute_id: data_dim, data_length
        estimators: Union[skbase.BaseEstimator, dict[str, skbase.BaseEstimator]],
        parameter_scaling: Optional[Callable[[Mu], Mu]] = None,
        training_data: Optional[
            tuple[Iterable[Mu], dict[str, Iterable[VectorSpace | np.ndarray]]]
        ] = None,
        shuffle_mode: str = "sort+shuffle",  # highly recommended, possible values are None, 'sort', 'shuffle', 'sort+shuffle'
        rom_name: str = "DataDrivenModel",
    ):
        assert isinstance(parameters, Parameters)
        assert isinstance(compute_shapes, dict)
        assert all(isinstance(k, str) for k in compute_shapes)
        self.reducable_computes = set(compute_shapes.keys())
        assert all(
            isinstance(v, tuple) and len(v) == 2 for v in compute_shapes.values()
        )
        assert all(
            isinstance(s, (VectorSpace, Number)) for s, _ in compute_shapes.values()
        )
        assert all(isinstance(c, Number) for _, c in compute_shapes.values())
        # We use skbase.clone to ensure no one tempers with the estimators while we hold them.
        if isinstance(estimators, skbase.BaseEstimator):
            estimators = {
                compute_id: skbase.clone(estimators)
                for compute_id in self.reducable_computes
            }
        assert isinstance(estimators, dict)
        assert all(isinstance(k, str) for k in estimators)
        assert all(k in self.reducable_computes for k in estimators)
        assert all(isinstance(v, skbase.BaseEstimator) for v in estimators.values())
        parameter_scaling = parameter_scaling or (lambda mu: mu)
        assert callable(parameter_scaling)
        # training_data will be checked in _parse_data
        # shuffle_mode will be checked in _parse_data
        assert isinstance(rom_name, str)
        rom_name = rom_name.strip()
        assert rom_name

        self.__auto_init(locals())
        self._data = self._parse_data(training_data)
        self._shuffle_data()

        # Keeping the fitted estimators is not required, but facilitates debugging and post-processing
        self._fitted_estimators = {}
        self._roms = {}

    def extend_training_data(self, training_inputs, training_states):
        additional_training_inputs, additional_training_data = self._parse_data(
            (training_inputs, training_states)
        )
        if len(additional_training_inputs) > 0:
            self._data[0] += additional_training_inputs
            for compute_id in self.reducable_computes:
                self._data[1][compute_id] += additional_training_data[compute_id]
        self._shuffle_data()

    def _shuffle_data(self):
        assert self.shuffle_mode in (None, "sort", "shuffle", "sort+shuffle")
        if self.shuffle_mode is None:
            return
        if len(self._data[0]) == 0:
            return
        data_tuples = list(
            zip(self._data[0], *(self._data[1][k] for k in self._data[1]))
        )
        if self.shuffle_mode in ("sort", "sort+shuffle"):
            if self.parameters.dim == 1:

                def reduce(pair):
                    return pair[0].to_numpy().ravel()[0]
            else:

                def reduce(pair):
                    return np.linalg.norm(pair[0].to_numpy())

            data_tuples.sort(key=reduce)
        if self.shuffle_mode in ("shuffle", "sort+shuffle"):
            get_rng().shuffle(data_tuples)
        reordered_inputs_and_outputs = list(zip(*data_tuples))
        reordered_inputs = reordered_inputs_and_outputs[0]
        reordered_outputs = {
            k: list(reordered_inputs_and_outputs[i + 1])
            for i, k in enumerate(self._data[1])
        }
        self._data = [reordered_inputs, reordered_outputs]

    def reconstruct(self, u, basis=None):
        return u

    def reduce(self, data_size: Optional[Number] = None) -> DataDrivenModel:
        """Fits the estimator to the amount of data specified by data_size."""
        if data_size is None:
            data_size = len(self._data[0])
        assert isinstance(data_size, Number)
        assert data_size >= 0
        assert data_size <= len(self._data[0])
        if data_size not in self._roms:
            self._fitted_estimators[data_size] = {
                compute_id: skbase.clone(estimator)
                for compute_id, estimator in self.estimators.items()
            }
            if data_size > 0:
                X = [mu.to_numpy() for mu in self._data[0][:data_size]]
                assert np.all(mu.shape == (self.parameters.dim,) for mu in X)
                X = np.stack(X)
                for compute_id, estimator in self._fitted_estimators[data_size].items():
                    Y = np.stack(self._data[1][compute_id][:data_size])
                    with self.logger.block(
                        f'Fitting estimator for {compute_id} to {"all" if data_size == len(self._data[0]) else "first"} {data_size} samples ...'
                    ):
                        estimator.fit(X, Y)
            self._roms[data_size] = DataDrivenModel(
                parameters=self.parameters,
                parameter_scaling=self.parameter_scaling,
                compute_shapes=self.compute_shapes,
                estimators=self._fitted_estimators[data_size],
                name=self.rom_name,
            )
            self._roms[data_size].disable_logging()
        return self._roms[data_size]

    def data_as_X_y(
        self, quantity=None, data_size: Number = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if quantity is None:
            assert len(self.reducable_computes) == 1
            quantity = next(iter(self.reducable_computes))
        assert quantity in self.reducable_computes
        if data_size is None:
            data_size = len(self._data[0])
        assert isinstance(data_size, Number)
        assert data_size >= 0
        assert data_size <= len(self._data[0])
        if data_size == 0:
            return np.empty((0, self.parameters.dim)), {
                compute_id: np.empty((0, data_length))
                for compute_id, (_, data_length) in self.compute_shapes.items()
            }
        X = [mu.to_numpy() for mu in self._data[0][:data_size]]
        assert np.all(mu.shape == (self.parameters.dim,) for mu in X)
        X = np.stack(X)
        y = np.stack(self._data[1][quantity][:data_size])
        return X, y

    def _parse_data(
        self,
        mus_and_data: Optional[
            tuple[Iterable[Mu], dict[str, Iterable[VectorSpace | np.ndarray]]]
        ] = None,
    ):
        parsed_new_data = [[], {compute_id: [] for compute_id in self.compute_shapes}]
        if mus_and_data is None:
            return parsed_new_data
        assert isinstance(mus_and_data, Iterable) and len(mus_and_data) == 2
        mus, data = mus_and_data
        if len(mus) == 0:
            return parsed_new_data
        assert isinstance(mus, Iterable)
        mus = [
            self.parameters.parse(mu) if not isinstance(mu, Mu) else mu for mu in mus
        ]
        mus = [self.parameter_scaling(mu) for mu in mus]
        parsed_data = {compute_id: [] for compute_id in self.compute_shapes}
        assert isinstance(data, dict)
        assert all(k in data for k in self.compute_shapes)
        # While we only want to add new data, we still parse everything as we check against existing entries
        for compute_id, (data_dim, data_length) in self.compute_shapes.items():
            computed_data = data[compute_id]
            assert isinstance(computed_data, Iterable)
            assert len(computed_data) == len(mus)
            if isinstance(data_dim, VectorSpace):
                assert all(isinstance(v, VectorArray) for v in computed_data)
                assert all(len(v) == data_length for v in computed_data)
                parsed_data[compute_id] = [v.to_numpy().ravel() for v in computed_data]
            else:
                assert isinstance(data_dim, Number) and data_dim >= 0
                assert all(isinstance(v, np.ndarray) for v in computed_data)
                computed_data = [
                    np.asarray(v).reshape(data_dim, data_length) for v in computed_data
                ]
                parsed_data[compute_id] = [np.asarray(v).ravel() for v in computed_data]
        for mu_index, mu in enumerate(mus):
            try:
                already_collected_mu_index = self._data[0].index(
                    mu
                )  # raise ValueError if mu not in self._data[0]
                for compute_id in self.compute_shapes:
                    new_data = parsed_data[compute_id][mu_index]
                    already_collected_data = self._data[1][compute_id][
                        already_collected_mu_index
                    ]
                    assert (
                        np.linalg.norm(already_collected_data - new_data, np.inf)
                        < 1e-14
                    )
            except ValueError:
                # index() raises for unknown mus, so we did not collect data for this one yet
                parsed_new_data[0].append(mu)
                for compute_id in self.compute_shapes:
                    parsed_new_data[1][compute_id].append(
                        parsed_data[compute_id][mu_index]
                    )

        return parsed_new_data

    def tune_hyperparameters(
        self,
        param_grid: dict,
        quantity: str = None,
        cv_folds: int = 5,
        scoring_norm: tuple = None,
    ):
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import GridSearchCV

        """Tune hyperparameters using GridSearchCV."""

        if quantity is None:
            assert len(self.reducable_computes) == 1
            quantity = next(iter(self.reducable_computes))

        # Use provided norm or default to l-inf
        if scoring_norm is None:
            scoring_norm = ("l_inf", lambda u: np.linalg.norm(u, np.inf))

        # Create custom scorer for relative error that matches validation exactly
        def relative_error(y_true, y_pred):
            """Compute relative error matching the validation metric."""
            errors = []
            for i in range(y_true.shape[0]):
                # Special handling for l_inf-l_2 norm
                if scoring_norm[0] == "l_inf-l_2":
                    # For CV, we only have access to numpy arrays, not VectorArrays
                    # So we interpret l_inf-l_2 as: for each parameter, compute L2 error
                    # Then take the max (L_inf) over all parameters
                    # This is what the validation does with VectorArrays
                    y_true_norm = np.linalg.norm(y_true[i], 2)
                    error_norm = np.linalg.norm(y_true[i] - y_pred[i], 2)
                else:
                    # Use provided function for other norms
                    y_true_norm = scoring_norm[1](y_true[i])
                    error_norm = scoring_norm[1](y_true[i] - y_pred[i])

                if y_true_norm > 0:
                    errors.append(error_norm / y_true_norm)
                else:
                    errors.append(0.0)  # Handle zero norm case

            # Return max error (L_inf over parameter space) to match validation
            return np.max(errors) if errors else 0.0

        scorer = make_scorer(relative_error, greater_is_better=False)

        # Get data for tuning
        X, y = self.data_as_X_y(quantity=quantity)

        # Ensure we're not using validation data in CV
        # Only use training data for hyperparameter tuning
        n_training = len(self.training_set) if hasattr(self, "training_set") else len(X)
        X_train = X[:n_training]
        y_train = y[:n_training]

        # Run grid search
        with self.logger.block(f"Tuning hyperparameters for {quantity}..."):
            search = GridSearchCV(
                estimator=self.estimators[quantity],
                param_grid=param_grid,
                scoring=scorer,
                cv=cv_folds,
                verbose=1,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            # Update estimator with best parameters
            self.estimators[quantity] = search.best_estimator_

            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best CV score: {-search.best_score_:.6f}")

            # Compute validation score on held-out data if available
            if hasattr(self, "validation_set") and n_training < len(X):
                X_val = X[n_training:]
                y_val = y[n_training:]
                val_errors = []
                y_val_pred = search.best_estimator_.predict(X_val)
                for i in range(y_val.shape[0]):
                    if scoring_norm[0] == "l_inf-l_2":
                        y_true_norm = np.linalg.norm(y_val[i], 2)
                        error_norm = np.linalg.norm(y_val[i] - y_val_pred[i], 2)
                    else:
                        y_true_norm = scoring_norm[1](y_val[i])
                        error_norm = scoring_norm[1](y_val[i] - y_val_pred[i])
                    if y_true_norm > 0:
                        val_errors.append(error_norm / y_true_norm)
                val_score = np.max(val_errors) if val_errors else 0.0
                self.logger.info(f"Validation score: {val_score:.6f}")

        return search
