# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import tempfile
import collections
import os
import subprocess
import time

from pymor.analyticalproblems.domaindescriptions import RectDomain, PolygonalDomain
from pymor.discretizers.builtin.grids.gmsh import load_gmsh
from pymor.core.exceptions import GmshMissing
from pymor.core.logger import getLogger


def discretize_gmsh(domain_description=None, geo_file=None, geo_file_path=None, msh_file_path=None,
                    mesh_algorithm='del2d', clscale=1., options='', refinement_steps=0):
    """Mesh a |DomainDescription| or an already existing Gmsh GEO-file using the Gmsh mesher.

    Parameters
    ----------
    domain_description
        A |DomainDescription| of the |PolygonalDomain| or |RectDomain| to discretize. Has to be
        `None` when `geo_file` is given.
    geo_file
        File handle of the Gmsh Geo-file to discretize. Has to be `None` when
        `domain_description` is given.
    geo_file_path
        Path of the created Gmsh GEO-file. When meshing a |PolygonalDomain| or |RectDomain| and
        `geo_file_path` is `None`, a temporary file will be created. If `geo_file` is specified,
        this is ignored and the path to `geo_file` will be used.
    msh_file_path
        Path of the created Gmsh MSH-file. If `None`, a temporary file will be created.
    mesh_algorithm
        The mesh generation algorithm to use (meshadapt, del2d, front2d).
    clscale
        Mesh element size scaling factor.
    options
        Other options to control the meshing procedure of Gmsh. See
        https://gmsh.info/doc/texinfo/gmsh.html#Command_002dline-options for all available
        options.
    refinement_steps
        Number of refinement steps to do after the initial meshing.

    Returns
    -------
    grid
        The generated :class:`~pymor.discretizers.builtin.grids.gmsh.GmshGrid`.
    boundary_info
        The generated :class:`~pymor.discretizers.builtin.grids.gmsh.GmshBoundaryInfo`.
    """
    assert domain_description is None or geo_file is None
    logger = getLogger('pymor.discretizers.builtin.domaindiscretizers.gmsh.discretize_gmsh')

    # run Gmsh; initial meshing
    logger.info('Checking for Gmsh ...')

    # when we are running MPI parallel and Gmsh is compiled with MPI support,
    # we have to make sure Gmsh does not notice the MPI environment or it will fail.
    env = {k: v for k, v in os.environ.items()
           if 'MPI' not in k.upper()}
    try:
        version = subprocess.check_output(['gmsh', '--version'], stderr=subprocess.STDOUT, env=env).decode()
    except (subprocess.CalledProcessError, OSError) as e:
        raise GmshMissing('Could not find Gmsh.'
                          ' Please ensure that the gmsh binary (https://gmsh.info/) is in your PATH.') from e

    logger.info('Found version ' + version.strip())

    def discretize_PolygonalDomain():
        # combine points and holes, since holes are points, too, and have to be stored as such.
        points = [domain_description.points]
        points.extend(domain_description.holes)

        return points, domain_description.boundary_types

    def discretize_RectDomain():
        points = [[domain_description.domain[0].tolist(),
                   [domain_description.domain[1][0], domain_description.domain[0][1]],
                   domain_description.domain[1].tolist(),
                   [domain_description.domain[0][0], domain_description.domain[1][1]]]]
        boundary_types = {domain_description.bottom: [1]}
        if domain_description.right not in boundary_types:
            boundary_types[domain_description.right] = [2]
        else:
            boundary_types[domain_description.right].append(2)
        if domain_description.top not in boundary_types:
            boundary_types[domain_description.top] = [3]
        else:
            boundary_types[domain_description.top].append(3)
        if domain_description.left not in boundary_types:
            boundary_types[domain_description.left] = [4]
        else:
            boundary_types[domain_description.left].append(4)

        if None in boundary_types:
            del boundary_types[None]

        return points, boundary_types

    # these two are referenced in a finally block, but were left undefined in some paths
    geo_file, msh_file = None, None
    try:
        # When a |PolygonalDomain| or |RectDomain| has to be discretized create a
        # Gmsh GE0-file and write all data.
        if domain_description is not None:
            logger.info('Writing Gmsh geometry file ...')
            # Create a temporary GEO-file if None is specified
            if geo_file_path is None:
                geo_file = tempfile.NamedTemporaryFile(mode='wt', delete=False, suffix='.geo')
                geo_file_path = geo_file.name
            else:
                geo_file = open(geo_file_path, 'w')

            if isinstance(domain_description, PolygonalDomain):
                points, boundary_types = discretize_PolygonalDomain()
            elif isinstance(domain_description, RectDomain):
                points, boundary_types = discretize_RectDomain()
            else:
                raise NotImplementedError(f'I do not know how to discretize {domain_description}')

            # assign ids to all points and write them to the GEO-file.
            for id, p in enumerate([p for ps in points for p in ps]):
                assert len(p) == 2
                geo_file.write('Point('+str(id+1)+') = '+str(p+[0, 0]).replace('[', '{').replace(']', '}')+';\n')

            # store points and their ids
            point_ids = dict(zip([str(p) for ps in points for p in ps],
                                 range(1, len([p for ps in points for p in ps])+1)))
            # shift points 1 entry to the left.
            points_deque = [collections.deque(ps) for ps in points]
            for ps_d in points_deque:
                ps_d.rotate(-1)
            # create lines by connecting the points with shifted points,
            # such that they form a polygonal chains.
            lines = [[point_ids[str(p0)], point_ids[str(p1)]]
                     for ps, ps_d in zip(points, points_deque) for p0, p1 in zip(ps, ps_d)]
            # assign ids to all lines and write them to the GEO-file.
            for l_id, l in enumerate(lines):
                geo_file.write('Line('+str(l_id+1)+')'+' = '+str(l).replace('[', '{').replace(']', '}')+';\n')

            # form line_loops (polygonal chains), create ids and write them to file.
            line_loops = [[point_ids[str(p)] for p in ps] for ps in points]
            line_loop_ids = range(len(lines)+1, len(lines)+len(line_loops)+1)
            for ll_id, ll in zip(line_loop_ids, line_loops):
                geo_file.write('Line Loop('+str(ll_id)+')'+' = '+str(ll).replace('[', '{').replace(']', '}')+';\n')

            # set this here explicitly for string conversion to make sense
            line_loop_ids = list(line_loop_ids)
            # create the surface defined by line loops,
            # starting with the exterior and then the holes.
            geo_file.write('Plane Surface(' + str(line_loop_ids[0]+1) + ')' + ' = '
                           + str(line_loop_ids).replace('[', '{').replace(']', '}') + ';\n')
            geo_file.write('Physical Surface("boundary") = {'+str(line_loop_ids[0]+1)+'};\n')

            # write boundaries.
            for boundary_type, bs in boundary_types.items():
                geo_file.write('Physical Line' + '("' + str(boundary_type) + '")' + ' = '
                               + str([l_id for l_id in bs]).replace('[', '{').replace(']', '}') + ';\n')

            geo_file.close()
        # When a GEO-File is provided just get the corresponding file path.
        else:
            geo_file_path = geo_file.name
        # Create a temporary MSH-file if no path is specified.
        if msh_file_path is None:
            msh_file = tempfile.NamedTemporaryFile(mode='wt', delete=False, suffix='.msh')
            msh_file_path = msh_file.name
            msh_file.close()

        tic = time.perf_counter()

        # run Gmsh; initial meshing
        logger.info('Calling Gmsh ...')
        cmd = ['gmsh', geo_file_path, '-2', '-algo', mesh_algorithm, '-clscale', str(clscale), options, '-o',
               msh_file_path]
        subprocess.check_call(cmd, env=env)

        # run gmsh; perform mesh refinement
        cmd = ['gmsh', msh_file_path, '-refine', '-o', msh_file_path]
        for i in range(refinement_steps):
            logger.info(f'Performing Gmsh refinement step {i+1}')
            subprocess.check_call(cmd, env=env)

        toc = time.perf_counter()
        t_gmsh = toc - tic
        logger.info(f'Gmsh took {t_gmsh} s')

        # Create |GmshGrid| and |GmshBoundaryInfo| form the just created MSH-file.
        grid, bi = load_gmsh(msh_file_path)
    finally:
        # delete tempfiles if they were created beforehand.
        if isinstance(geo_file, tempfile._TemporaryFileWrapper):
            os.remove(geo_file_path)
        if isinstance(msh_file, tempfile._TemporaryFileWrapper):
            os.remove(msh_file_path)

    return grid, bi
