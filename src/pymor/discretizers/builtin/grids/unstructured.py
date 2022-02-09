# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.discretizers.builtin.grids.interfaces import Grid
from pymor.discretizers.builtin.grids.referenceelements import triangle


class UnstructuredTriangleGrid(Grid):
    """A generic unstructured, triangular grid.

    Use :meth:`~UnstructuredTriangleGrid.from_vertices` to instantiate
    the grid from vertex coordinates and connectivity data.
    """

    dim = 2
    reference_element = triangle

    def __init__(self, sizes, subentity_data, embedding_data):
        self.__auto_init(locals())
        vertices = self.centers(2)
        self.domain = np.array([[np.min(vertices[:, 0]), np.min(vertices[:, 1])],
                                [np.max(vertices[:, 0]), np.max(vertices[:, 1])]])

    @classmethod
    def from_vertices(cls, vertices, faces):
        """Instantiate grid from vertex coordinates and connectivity data.

        Parameters
        ----------
        vertices
            A (num_vertices, 2)-shaped |array| containing the coordinates
            of all vertices in the grid. The row numbers in the array will
            be the global indices of the given vertices (codim 2 entities).
        faces
            A (num_faces, 3)-shaped |array| containing the global indices
            of the vertices which define a given triangle in the grid.
            The row numbers in the array will be the global indices of the
            given triangles (codim 0 entities).
        """
        assert faces.shape[1] == 3
        assert np.min(faces) == 0
        assert np.max(faces) == len(vertices) - 1

        vertices = vertices.astype(np.float64, copy=False)
        faces = faces.astype(np.int32, copy=False)
        edges, num_edges = compute_edges(faces)

        COORDS = vertices[faces]
        SHIFTS = COORDS[:, 0, :]
        TRANS = COORDS[:, 1:, :] - SHIFTS[:, np.newaxis, :]
        TRANS = TRANS.swapaxes(1, 2)

        sizes = (len(faces), num_edges, len(vertices))
        subentity_data = (np.arange(len(faces), dtype=np.int32).reshape(-1, 1), edges, faces)
        embedding_data = (TRANS, SHIFTS)

        return cls(sizes, subentity_data, embedding_data)

    def size(self, codim=0):
        assert 0 <= codim <= 2, 'Invalid codimension'
        return self.sizes[codim]

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 2, 'Invalid codimension'
        if subentity_codim is None:
            subentity_codim = codim + 1
        assert codim <= subentity_codim <= self.dim, 'Invalid subentity codimensoin'
        if codim == 0:
            return self.subentity_data[subentity_codim]
        else:
            return super().subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self.embedding_data
        else:
            return super().embeddings(codim)

    def visualize(self, U, codim=2, **kwargs):
        """Visualize scalar data associated to the grid as a patch plot.

        Parameters
        ----------
        U
            |NumPy array| of the data to visualize. If `U.dim == 2 and len(U) > 1`, the
            data is visualized as a time series of plots. Alternatively, a tuple of
            |Numpy arrays| can be provided, in which case a subplot is created for
            each entry of the tuple. The lengths of all arrays have to agree.
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 2).
        kwargs
            See :func:`~pymor.discretizers.builtin.gui.visualizers.PatchVisualizer.visualize`
        """
        from pymor.discretizers.builtin.gui.visualizers import PatchVisualizer
        from pymor.vectorarrays.interface import VectorArray
        from pymor.vectorarrays.numpy import NumpyVectorSpace
        if isinstance(U, (np.ndarray, VectorArray)):
            U = (U,)
        assert all(isinstance(u, (np.ndarray, VectorArray)) for u in U)
        U = tuple(NumpyVectorSpace.make_array(u) if isinstance(u, np.ndarray) else
                  u if isinstance(u.space, NumpyVectorSpace) else
                  NumpyVectorSpace.make_array(u.to_numpy())
                  for u in U)
        PatchVisualizer(self, codim=codim).visualize(U, **kwargs)

    def __str__(self):
        return 'UnstructuredTriangleGrid with {} triangles, {} edges, {} vertices'.format(*self.sizes)


def compute_edges(subentities):
    X = np.empty_like(subentities, dtype=[('l', np.int32), ('h', np.int32)])

    X['l'][:, 0] = np.min(subentities[:, 1:3], axis=1)
    X['l'][:, 1] = np.min(subentities[:, 0:3:2], axis=1)
    X['l'][:, 2] = np.min(subentities[:, 0:2], axis=1)
    X['h'][:, 0] = np.max(subentities[:, 1:3], axis=1)
    X['h'][:, 1] = np.max(subentities[:, 0:3:2], axis=1)
    X['h'][:, 2] = np.max(subentities[:, 0:2], axis=1)

    U, I = np.unique(X, return_inverse=True)
    return I.reshape(subentities.shape).astype(np.int32), len(U)
