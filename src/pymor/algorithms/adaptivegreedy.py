# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from fractions import Fraction

import numpy as np

from pymor.core.interfaces import BasicInterface
from pymor.parameters.base import Parameter
from pymor.parameters.spaces import CubicParameterSpace


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
        values = self.ranges[:, 0] + self.dimensions * map(float, vertex)
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
            zip(*((node.level, node.center, node.vertex_ids, node.creation_time) for node in self._iter_leafs()))
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
