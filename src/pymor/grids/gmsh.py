# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from collections import defaultdict

import numpy as np
import time

from pymor.core.exceptions import GmshError
from pymor.core.logger import getLogger
from pymor.grids.interfaces import BoundaryInfoInterface
from pymor.grids.unstructured import UnstructuredTriangleGrid


def load_gmsh(gmsh_file):
    """Parse a Gmsh file and create a corresponding :class:`GmshGrid` and :class:`GmshBoundaryInfo`.

    Parameters
    ----------
    gmsh_file
        File handle of the Gmsh MSH-file.

    Returns
    -------
    grid
        The generated :class:`GmshGrid`.
    boundary_info
        The generated :class:`GmshBoundaryInfo`.
    """
    logger = getLogger('pymor.grids.gmsh.load_gmsh')

    logger.info('Parsing gmsh file ...')
    tic = time.time()
    sections = _parse_gmsh_file(gmsh_file)
    toc = time.time()
    t_parse = toc - tic

    logger.info('Create GmshGrid ...')
    tic = time.time()
    grid = GmshGrid(sections)
    toc = time.time()
    t_grid = toc - tic

    logger.info('Create GmshBoundaryInfo ...')
    tic = time.time()
    bi = GmshBoundaryInfo(grid, sections)
    toc = time.time()
    t_bi = toc - tic

    logger.info(f'Parsing took {t_parse} s; Grid creation took {t_grid} s; BoundaryInfo creation took {t_bi} s')

    return grid, bi


class GmshGrid(UnstructuredTriangleGrid):
    """An :class:`~pymor.grids.unstructured.UnstructuredTriangleGrid` built from an existing Gmsh MSH-file.

    Parameters
    ----------
    sections
        Parsed sections of the MSH-file as returned by :func:`load_gmsh`.
    """

    def __init__(self, sections):
        self.logger.info('Checking if grid is a 2d triangular grid ...')
        assert {'Nodes', 'Elements', 'PhysicalNames'} <= set(sections.keys())
        assert set(sections['Elements'].keys()) <= {'line', 'triangle'}
        assert 'triangle' in sections['Elements']
        assert all(n[1][2] == 0 for n in sections['Nodes'])

        node_ids = dict(zip([n[0] for n in sections['Nodes']], np.arange(len(sections['Nodes']), dtype=np.int32)))
        vertices = np.array([n[1][0:2] for n in sections['Nodes']])

        faces = np.array([[node_ids[nodes[0]], node_ids[nodes[1]], node_ids[nodes[2]]]
                         for _, _, nodes in sections['Elements']['triangle']])
        super().__init__(vertices, faces)

    def __str__(self):
        return f'GmshGrid with {self.size(0)} triangles, {self.size(1)} edges, {self.size(2)} vertices'


class GmshBoundaryInfo(BoundaryInfoInterface):
    """|BoundaryInfo| for a :class:`GmshGrid`.

    Parameters
    ----------
    grid
        The corresponding :class:`GmshGrid`.
    sections
        Parsed sections of the MSH-file as returned by :func:`load_gmsh`.
    """

    def __init__(self, grid, sections):
        assert isinstance(grid, GmshGrid)
        self.grid = grid

        # Save boundary types.
        self.boundary_types = [pn[2] for pn in sections['PhysicalNames'] if pn[1] == 1]

        # Compute ids, since Gmsh starts numbering with 1 instead of 0.
        name_ids = dict(zip([pn[0] for pn in sections['PhysicalNames']], np.arange(len(sections['PhysicalNames']),
                                                                                   dtype=np.int32)))
        node_ids = dict(zip([n[0] for n in sections['Nodes']], np.arange(len(sections['Nodes']), dtype=np.int32)))

        if 'line' in sections['Elements']:
            superentities = grid.superentities(2, 1)

            # find the edge for given vertices.
            def find_edge(vertices):
                edge_set = set(superentities[vertices[0]]).intersection(superentities[vertices[1]]) - {-1}
                if len(edge_set) != 1:
                    raise ValueError
                return next(iter(edge_set))

            line_ids = {l[0]: find_edge([node_ids[l[2][0]], node_ids[l[2][1]]]) for l in sections['Elements']['line']}

        # compute boundary masks for all boundary types.
        masks = {}
        for bt in self.boundary_types:
            masks[bt] = [np.array([False]*grid.size(1)), np.array([False]*grid.size(2))]
            masks[bt][0][[line_ids[l[0]] for l in sections['Elements']['line']]] = \
                [(bt == sections['PhysicalNames'][name_ids[l[1][0]]][2]) for l in sections['Elements']['line']]
            ind = np.array([node_ids[n] for l in sections['Elements']['line'] for n in l[2]])
            val = masks[bt][0][[line_ids[l[0]] for l in sections['Elements']['line'] for n in l[2]]]
            masks[bt][1][ind[val]] = True

        self._masks = masks

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        assert boundary_type in self.boundary_types
        return self._masks[boundary_type][codim - 1]


def _parse_gmsh_file(f):

    allowed_sections = ['Nodes', 'Elements', 'PhysicalNames', 'Periodic', 'NodeData',
                        'ElementData', 'ElementNodeData']

    supported_sections = ['Nodes', 'Elements', 'PhysicalNames']

    try:

        l = next(f).strip()
        if l != '$MeshFormat':
            raise GmshError(f'expected $MeshFormat, got {l}')

        l = next(f).strip()
        header = l.split(' ')
        if len(header) != 3:
            raise GmshError(f'header {l} has {len(header)} fields, expected 3')

        if header[0] != '2.2':
            raise GmshError(f'wrong file format version: got {header[0]}, expected 2.2')

        try:
            file_type = int(header[1])
        except ValueError:
            raise GmshError(f'malformed header: expected integer, got {header[1]}')

        if file_type != 0:
            raise GmshError('wrong file type: only ASCII gmsh files are supported')

        try:
            data_size = int(header[2])    # NOQA
        except ValueError:
            raise GmshError(f'malformed header: expected integer, got {header[2]}')

        l = next(f).strip()
        if l != '$EndMeshFormat':
            raise GmshError(f'expected $EndMeshFormat, got {l}')

    except StopIteration:
        raise GmshError('unexcpected end of file')

    in_section = False
    sections = defaultdict(list)

    for l in f:
        l = l.strip()
        if l == '':
            continue
        if not in_section:
            if not l.startswith('$'):
                raise GmshError(f'expected section name, got {l}')
            section = l[1:]
            if section not in allowed_sections:
                raise GmshError(f'unknown section type: {section}')
            if section not in supported_sections:
                raise GmshError(f'unsupported section type: {section}')
            if section in sections:
                raise GmshError(f'only one {section} section allowed')
            in_section = True
        elif l.startswith('$'):
            if l != '$End' + section:
                raise GmshError(f'expected $End{section}, got {l}')
            in_section = False
        else:
            sections[section].append(l)

    if in_section:
        raise GmshError(f'file ended while in section {section}')

    # now we parse each section ...

    def parse_nodes(nodes):
        try:
            num_nodes = int(nodes[0])
        except ValueError:
            raise GmshError(f'first line of nodes sections is not a number: {nodes[0]}')
        if len(nodes) != num_nodes + 1:
            raise GmshError('number-of-nodes field does not match number of lines in nodes section')

        nodes = [n.split(' ') for n in nodes[1:]]
        if not all(len(n) == 4 for n in nodes):
            raise GmshError('malformed nodes section')

        try:
            nodes = [(int(a), (float(b), float(c), float(d))) for a, b, c, d in nodes]
        except ValueError:
            raise GmshError('malformed nodes section')

        return nodes

    def parse_elements(elements):
        try:
            num_elements = int(elements[0])
        except ValueError:
            raise GmshError(f'first line of elements sections is not a number: {elements[0]}')
        if len(elements) != num_elements + 1:
            raise GmshError('number-of-elements field does not match number of lines in elements section')

        elements = [e.split(' ') for e in elements[1:]]
        try:
            elements = [tuple(int(f) for f in e) for e in elements]
        except ValueError:
            raise GmshError('malformed elements section')

        element_types = {1: 'line', 2: 'triangle'}
        element_nodes = {'line': 2, 'triangle': 3}

        def parse_line(fields):
            if fields[1] not in element_types:
                raise GmshError(f'element type {fields[0]} not supported')
            element_type = element_types[fields[1]]
            num_nodes = element_nodes[element_type]
            num_tags = fields[2]
            if len(fields) != num_nodes + num_tags + 3:
                raise GmshError('malformed elements section')
            return element_type, (fields[0], tuple(fields[3:3 + num_tags]), fields[3 + num_tags:])

        elements_by_type = defaultdict(list)
        for e in elements:
            t, l = parse_line(e)
            elements_by_type[t].append(l)

        return elements_by_type

    def parse_names(physical_names):
        try:
            num_elements = int(physical_names[0])
        except ValueError:
            raise GmshError(f'first line of physical names sections is not a number: {physical_names[0]}')
        if len(physical_names) != num_elements + 1:
            raise GmshError('number-of-names field does not match number of lines in physical names section')

        physical_names = [pn.split(' ') for pn in physical_names[1:]]
        if not all(len(pn) == 3 for pn in physical_names):
            raise GmshError('malformed physical names section')

        try:
            physical_names = [(int(b), int(a), str(c).replace('"', '')) for a, b, c in physical_names]
        except ValueError:
            raise GmshError('malformed physical names section')

        return physical_names

    parser_map = {'Nodes': parse_nodes, 'Elements': parse_elements, 'PhysicalNames': parse_names}

    for k, v in sections.items():
        sections[k] = parser_map[k](v)

    return sections
