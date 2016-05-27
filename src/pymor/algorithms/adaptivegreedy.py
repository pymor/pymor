# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from fractions import Fraction

import numpy as np
import time

from pymor.algorithms.basisextension import gram_schmidt_basis_extension
from pymor.core.exceptions import ExtensionError
from pymor.core.interfaces import BasicInterface
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.manager import RemoteObjectManager
from pymor.parameters.base import Parameter
from pymor.parameters.spaces import CubicParameterSpace


def adaptive_greedy(discretization, reductor, parameter_space=None,
                    initial_basis=None, use_estimator=True, error_norm=None,
                    extension_algorithm=gram_schmidt_basis_extension, target_error=None, max_extensions=None,
                    validation_mus=0, rho=1.1, gamma=0.2, theta=0.,
                    visualize=False, visualize_vertex_size=80, pool=dummy_pool):
    """Greedy basis generation algorithm with adaptively refined training set.

    This method extends pyMOR's default :func:`~pymor.algorithms.greedy.greedy`
    greedy basis generation algorithm by adaptive refinement of the
    parameter training set according to [HDO11]_ to prevent overfitting
    of the reduced basis to the training set. This is achieved by
    estimating the reduction error on an additional validation set of
    parameters. If the ratio between the estimated errors on the validation
    set and the validation set is larger than `rho`, the training set
    is refined using standard grid refinement techniques.

    .. [HDO11] Haasdonk, B.; Dihlmann, M. & Ohlberger, M.,
               A training set and multiple bases generation approach for
               parameterized model reduction based on adaptive grids in
               parameter space,
               Math. Comput. Model. Dyn. Syst., 2011, 17, 423-442

    Parameters
    ----------
    discretization
        See :func:`~pymor.algorithms.greedy.greedy`.
    reductor
        See :func:`~pymor.algorithms.greedy.greedy`.
    parameter_space
        The |ParameterSpace| for which to compute the reduced model. If `None`
        the parameter space of the `discretization` is used.
    initial_basis
        See :func:`~pymor.algorithms.greedy.greedy`.
    use_estimator
        See :func:`~pymor.algorithms.greedy.greedy`.
    error_norm
        See :func:`~pymor.algorithms.greedy.greedy`.
    extension_algorithm
        See :func:`~pymor.algorithms.greedy.greedy`.
    target_error
        See :func:`~pymor.algorithms.greedy.greedy`.
    max_extensions
        See :func:`~pymor.algorithms.greedy.greedy`.
    validation_mus
        One of the following:
          - a list of |Parameters| to use as validation set,
          - a positive number indicating the number of random parameters
            to use as validation set,
          - a non-positive number, indicating the negative number of random
            parameters to use as validation set in addition to the centers
            of the elements of the adaptive training set.
    rho
        Maximum allowed ratio between maximum estimated error on validation
        set vs. maximum estimated error on training set. If the ratio is
        larger, the training set is refined.
    gamma
        Weight of the age penalty term of the training set refinement
        indicators.
    theta
        Ratio of training set elements to select for refinement.
        (One element is always refined.)
    visualize
        If `True`, visualize the refinement indicators. (Only available
        for 2 and 3 dimensional parameter spaces.)
    visualize_vertex_size
        Size of the vertices in the visualization.
    pool
        See :func:`~pymor.algorithms.greedy.greedy`.

    Returns
    -------
    Dict with the following fields:

        :basis:                  The reduced basis.
        :reduced_discretization: The reduced |Discretization| obtained for the
                                 computed basis.
        :reconstructor:          Reconstructor for `reduced_discretization`.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :max_val_errs:           Sequence of maximum errors on the validation set.
        :max_val_err_mus:        The parameters corresponding to `max_val_errs`.
        :refinements:            Number of refinements made in each extension step.
        :training_set_sizes:     The final size of the training set in each extension step.
    """

    def estimate(mus):
        if use_estimator:
            errors = pool.map(_estimate, mus, rd=rd)
        else:
            errors = pool.map(_estimate, mus, rd=rd, d=d, rc=rc, error_norm=error_norm)
        # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
        # if necessary
        return np.array([x[0] if hasattr(x, '__len__') else x for x in errors])

    logger = getLogger('pymor.algorithms.adaptivegreedy.adaptive_greedy')

    if pool is None or pool is dummy_pool:
        pool = dummy_pool
    else:
        logger.info('Using pool of {} workers for parallel greedy search'.format(len(pool)))

    with RemoteObjectManager() as rom:
        # Push everything we need during the greedy search to the workers.
        if not use_estimator:
            rom.manage(pool.push(discretization))
            if error_norm:
                rom.manage(pool.push(error_norm))

        tic = time.time()

        # initial setup for main loop
        d = discretization
        basis = initial_basis
        rd, rc, reduction_data = None, None, None
        hierarchic = False

        # setup training and validation sets
        parameter_space = parameter_space or d.parameter_space
        sample_set = AdaptiveSampleSet(parameter_space)
        if validation_mus <= 0:
            validation_set = sample_set.center_mus + parameter_space.sample_randomly(-validation_mus)
        else:
            validation_set = parameter_space.sample_randomly(validation_mus)
        if visualize and sample_set.dim not in (2, 3):
            raise NotImplementedError
        logger.info('Training set size: {}. Validation set size: {}'
                    .format(len(sample_set.vertex_mus), len(validation_set)))

        extensions = 0
        max_errs = []
        max_err_mus = []
        max_val_errs = []
        max_val_err_mus = []
        refinements = []
        training_set_sizes = []

        while True:  # main loop
            with logger.block('Reducing ...'):
                rd, rc, reduction_data = reductor(discretization, basis) if not hierarchic \
                    else reductor(discretization, basis, extends=(rd, rc, reduction_data))

            current_refinements = 0
            while True:  # estimate reduction errors and refine training set until no overfitting is detected

                # estimate on training set
                with logger.block('Estimating errors ...'):
                    errors = estimate(sample_set.vertex_mus)
                max_err_ind = np.argmax(errors)
                max_err, max_err_mu = errors[max_err_ind], sample_set.vertex_mus[max_err_ind]
                logger.info('Maximum error after {} extensions: {} (mu = {})'.format(extensions, max_err, max_err_mu))

                # estimate on validation set
                val_errors = estimate(validation_set)
                max_val_err_ind = np.argmax(val_errors)
                max_val_err, max_val_err_mu = val_errors[max_val_err_ind], validation_set[max_val_err_ind]
                logger.info('Maximum validation error: {}'.format(max_val_err))
                logger.info('Validation error to training error ratio: {:.3e}'.format(max_val_err / max_err))

                if max_val_err >= max_err * rho:  # overfitting?

                    # compute element indicators for training set refinement
                    if current_refinements == 0:
                        logger.info2('Overfitting detected. Computing element indicators ...')
                    else:
                        logger.info3('Overfitting detected after refinement. Computing element indicators ...')
                    vertex_errors = np.max(errors[sample_set.vertex_ids], axis=1)
                    center_errors = estimate(sample_set.center_mus)
                    indicators_age_part = (gamma * sample_set.volumes / sample_set.total_volume
                                           * (sample_set.refinement_count - sample_set.creation_times))
                    indicators_error_part = np.max([vertex_errors, center_errors], axis=0) / max_err
                    indicators = indicators_age_part + indicators_error_part

                    # select elements
                    sorted_indicators_inds = np.argsort(indicators)[::-1]
                    refinement_elements = sorted_indicators_inds[:max(int(len(sorted_indicators_inds) * theta), 1)]
                    logger.info('Refining {} elements: {}'.format(len(refinement_elements), refinement_elements))

                    # visualization
                    if visualize:
                        from mpl_toolkits.mplot3d import Axes3D  # NOQA
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.subplot(2, 2, 1, projection=None if sample_set.dim == 2 else '3d')
                        plt.title('estimated errors')
                        sample_set.visualize(vertex_data=errors, center_data=center_errors, new_figure=False)
                        plt.subplot(2, 2, 2, projection=None if sample_set.dim == 2 else '3d')
                        plt.title('indicators_error_part')
                        vmax = np.max([indicators_error_part, indicators_age_part, indicators])
                        data = {('volume_data' if sample_set.dim == 2 else 'center_data'): indicators_error_part}
                        sample_set.visualize(vertex_size=visualize_vertex_size, vmin=0, vmax=vmax, new_figure=False,
                                             **data)
                        plt.subplot(2, 2, 3, projection=None if sample_set.dim == 2 else '3d')
                        plt.title('indicators_age_part')
                        data = {('volume_data' if sample_set.dim == 2 else 'center_data'): indicators_age_part}
                        sample_set.visualize(vertex_size=visualize_vertex_size, vmin=0, vmax=vmax, new_figure=False,
                                             **data)
                        plt.subplot(2, 2, 4, projection=None if sample_set.dim == 2 else '3d')
                        if sample_set.dim == 2:
                            plt.title('indicators')
                            sample_set.visualize(volume_data=indicators,
                                                 center_data=np.zeros(len(refinement_elements)),
                                                 center_inds=refinement_elements,
                                                 vertex_size=visualize_vertex_size, vmin=0, vmax=vmax, new_figure=False)
                        else:
                            plt.title('selected cells')
                            sample_set.visualize(center_data=np.zeros(len(refinement_elements)),
                                                 center_inds=refinement_elements,
                                                 vertex_size=visualize_vertex_size, vmin=0, vmax=vmax, new_figure=False)
                        plt.show()

                    # refine training set
                    sample_set.refine(refinement_elements)
                    current_refinements += 1

                    # update validation set if needed
                    if validation_mus <= 0:
                        validation_set = sample_set.center_mus + parameter_space.sample_randomly(-validation_mus)

                    logger.info('New training set size: {}. New validation set size: {}'
                                .format(len(sample_set.vertex_mus), len(validation_set)))
                    logger.info('Number of refinements: {}'.format(sample_set.refinement_count))
                    logger.info('')
                else:
                    break  # no overfitting, leave the refinement loop

            max_errs.append(max_err)
            max_err_mus.append(max_err_mu)
            max_val_errs.append(max_val_err)
            max_val_err_mus.append(max_val_err_mu)
            refinements.append(current_refinements)
            training_set_sizes.append(len(sample_set.vertex_mus))

            # break if traget error reached
            if target_error is not None and max_err <= target_error:
                logger.info('Reached maximal error on snapshots of {} <= {}'.format(max_err, target_error))
                break

            # basis extension
            with logger.block('Computing solution snapshot for mu = {} ...'.format(max_err_mu)):
                U = discretization.solve(max_err_mu)
            with logger.block('Extending basis with solution snapshot ...'):
                try:
                    basis, extension_data = extension_algorithm(basis, U)
                except ExtensionError:
                    logger.info('Extension failed. Stopping now.')
                    break
            extensions += 1
            if 'hierarchic' not in extension_data:
                logger.warn('Extension algorithm does not report if extension was hierarchic. Assuming it was\'nt ..')
                hierarchic = False
            else:
                hierarchic = extension_data['hierarchic']

            logger.info('')

            # break if prescribed basis size reached
            if max_extensions is not None and extensions >= max_extensions:
                logger.info('Maximum number of {} extensions reached.'.format(max_extensions))
                with logger.block('Reducing once more ...'):
                    rd, rc, reduction_data = reductor(discretization, basis) if not hierarchic \
                        else reductor(discretization, basis, extends=(rd, rc, reduction_data))
                break

    tictoc = time.time() - tic
    logger.info('Greedy search took {} seconds'.format(tictoc))
    return {'basis': basis, 'reduced_discretization': rd, 'reconstructor': rc,
            'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions,
            'max_val_errs': max_val_errs, 'max_val_err_mus': max_val_err_mus,
            'refinements': refinements, 'training_set_sizes': training_set_sizes,
            'time': tictoc, 'reduction_data': reduction_data}


def _estimate(mu, rd=None, d=None, rc=None, error_norm=None):
    """Called by :func:`adaptive_greedy`."""
    if d is None:
        return rd.estimate(rd.solve(mu), mu)
    elif error_norm is not None:
        return error_norm(d.solve(mu) - rc.reconstruct(rd.solve(mu)))
    else:
        return (d.solve(mu) - rc.reconstruct(rd.solve(mu))).l2_norm()


class AdaptiveSampleSet(BasicInterface):
    """An adaptive parameter samples set.

    Used by :func:`adaptive_greedy`.
    """

    def __init__(self, parameter_space):
        assert isinstance(parameter_space, CubicParameterSpace)
        self.parameter_space = parameter_space
        self.parameter_type = parameter_space.parameter_type
        self.ranges = np.concatenate([np.tile(np.array(parameter_space.ranges[k])[np.newaxis, :],
                                              [np.prod(shape), 1])
                                      for k, shape in parameter_space.parameter_type.items()], axis=0)
        self.dimensions = self.ranges[:, 1] - self.ranges[:, 0]
        self.total_volume = np.prod(self.dimensions)
        self.dim = len(self.dimensions)
        self._vertex_to_id_map = {}
        self.vertices = []
        self.vertex_mus = []
        self.refinement_count = 0
        self.element_tree = self.Element(0, (Fraction(1, 2),) * self.dim, self)
        self._update()

    def refine(self, ids):
        self.refinement_count += 1
        leafs = [node for i, node in enumerate(self._iter_leafs()) if i in ids]
        for node in leafs:
            node.refine(self)
        self._update()

    def map_vertex_to_mu(self, vertex):
        values = self.ranges[:, 0] + self.dimensions * list(map(float, vertex))
        mu = Parameter({})
        for k, shape in self.parameter_type.items():
            count = np.prod(shape)
            head, values = values[:count], values[count:]
            mu[k] = np.array(head).reshape(shape)
        return mu

    def visualize(self, vertex_data=None, vertex_inds=None, center_data=None, center_inds=None, volume_data=None,
                  vertex_size=80, vmin=None, vmax=None, new_figure=True):
        if self.dim not in (2, 3):
            raise ValueError('Can only visualize samples of dimension 2, 3')

        vertices = np.array(self.vertices).astype(float) * self.dimensions[np.newaxis, :] + self.ranges[:, 0]
        centers = np.array(self.centers).astype(float) * self.dimensions[np.newaxis, :] + self.ranges[:, 0]
        if vmin is None:
            vmin = np.inf
            if vertex_data is not None:
                vmin = min(vmin, np.min(vertex_data))
            if center_data is not None:
                vmin = min(vmin, np.min(center_data))
            if volume_data is not None:
                vmin = min(vmin, np.min(volume_data))

        if vmax is None:
            vmax = -np.inf
            if vertex_data is not None:
                vmax = max(vmax, np.max(vertex_data))
            if center_data is not None:
                vmax = max(vmax, np.max(center_data))
            if volume_data is not None:
                vmax = max(vmax, np.max(volume_data))

        if self.dim == 2:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle
            if new_figure:
                plt.figure()
            plt.xlim(self.ranges[0])
            plt.ylim(self.ranges[1])

            # draw volumes
            rects = []
            for leaf, level in zip(self.vertex_ids, self.levels):
                size = 1. / 2**level
                ll = self.vertices[leaf[0]] * self.dimensions + self.ranges[:, 0]
                rects.append(Rectangle(ll, size * self.dimensions[0], size * self.dimensions[1],
                                       facecolor='white', zorder=-1))
            if volume_data is not None:
                coll = PatchCollection(rects, match_original=False)
                coll.set_array(volume_data)
                coll.set_clim(vmin, vmax)
            else:
                coll = PatchCollection(rects, match_original=True)
            plt.gca().add_collection(coll)
            plt.sci(coll)

            # draw vertex data
            if vertex_data is not None:
                vtx = vertices[vertex_inds] if vertex_inds is not None else vertices
                plt.scatter(vtx[:, 0], vtx[:, 1], c=vertex_data, vmin=vmin, vmax=vmax, s=vertex_size)

            # draw center data
            if center_data is not None:
                cts = centers[center_inds] if center_inds is not None else centers
                plt.scatter(cts[:, 0], cts[:, 1], c=center_data, vmin=vmin, vmax=vmax, s=vertex_size)

            if volume_data is not None or center_data is not None or vertex_data is not None:
                plt.colorbar()
            if new_figure:
                plt.show()

        elif self.dim == 3:
            if volume_data is not None:
                raise NotImplementedError

            cube = np.array([[0., 0., 0.],
                             [1., 0., 0.],
                             [1., 1., 0.],
                             [0., 1., 0.],
                             [0., 0., 0.],
                             [0., 0., 1.],
                             [1., 0., 1.],
                             [1., 0., 0.],
                             [1., 0., 1.],
                             [1., 1., 1.],
                             [1., 1., 0.],
                             [1., 1., 1.],
                             [0., 1., 1.],
                             [0., 1., 0.],
                             [0., 1., 1.],
                             [0., 0., 1.]])

            from mpl_toolkits.mplot3d import Axes3D  # NOQA
            import matplotlib.pyplot as plt
            if new_figure:
                plt.figure()
                plt.gca().add_subplot(111, projection='3d')
            ax = plt.gca()

            # draw cells
            for leaf, level in zip(self.vertex_ids, self.levels):
                size = 1. / 2**level
                ll = self.vertices[leaf[0]] * self.dimensions + self.ranges[:, 0]
                c = cube * self.dimensions * size + ll
                ax.plot3D(c[:, 0], c[:, 1], c[:, 2], color='lightgray', zorder=-1)

            p = None
            # draw vertex data
            if vertex_data is not None:
                vtx = vertices[vertex_inds] if vertex_inds is not None else vertices
                p = ax.scatter(vtx[:, 0], vtx[:, 1], vtx[:, 2],
                               c=vertex_data, vmin=vmin, vmax=vmax, s=vertex_size)

            # draw center data
            if center_data is not None:
                cts = centers[center_inds] if center_inds is not None else centers
                p = ax.scatter(cts[:, 0], cts[:, 1], cts[:, 2],
                               c=center_data, vmin=vmin, vmax=vmax, s=vertex_size)

            if p is not None:
                plt.colorbar(p)
            if new_figure:
                plt.show()

        else:
            assert False

    def _iter_leafs(self):
        def walk(node):
            if node.children:
                for node in node.children:
                    for leaf in walk(node):
                        yield leaf
            else:
                yield node

        return walk(self.element_tree)

    def _update(self):
        self.levels, self.centers, vertex_ids, creation_times = \
            list(zip(*((node.level, node.center, node.vertex_ids, node.creation_time) for node in self._iter_leafs())))
        self.levels = np.array(self.levels)
        self.volumes = self.total_volume / ((2**self.dim)**self.levels)
        self.vertex_ids = np.array(vertex_ids)
        self.center_mus = [self.map_vertex_to_mu(v) for v in self.centers]
        self.creation_times = np.array(creation_times)

    def _add_vertex(self, v):
        v_id = self._vertex_to_id_map.get(v)
        if v_id is None:
            v_id = len(self.vertices)
            self.vertices.append(v)
            self.vertex_mus.append(self.map_vertex_to_mu(v))
            self._vertex_to_id_map[v] = v_id
        return v_id

    class Element(object):
        __slots__ = ['level', 'center', 'vertex_ids', 'children', 'creation_time']

        def __init__(self, level, center, sample_set):
            self.level, self.center, self.creation_time = level, center, sample_set.refinement_count
            vertex_ids = []
            lower_corner = [x - Fraction(1, 2**(level + 1)) for x in center]
            for x in range(2**len(center)):
                v = list(lower_corner)
                for d in range(len(center)):
                    y, x = x % 2, x // 2
                    if y:
                        v[d] += Fraction(1, 2**level)
                vertex_ids.append(sample_set._add_vertex(tuple(v)))

            self.vertex_ids = vertex_ids
            self.children = []

        def refine(self, sample_set):
            level = self.level
            center = self.center
            for x in range(2**len(center)):
                v = list(center)
                for d in range(len(center)):
                    y, x = x % 2, x // 2
                    v[d] += Fraction(1, 2**(level+2)) * (y * 2 - 1)
                self.children.append(AdaptiveSampleSet.Element(level + 1, tuple(v), sample_set))
