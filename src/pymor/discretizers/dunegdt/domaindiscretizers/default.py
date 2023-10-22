
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_DUNEGDT:

    import math as m
    from numbers import Number

    from pymor.analyticalproblems.domaindescriptions import (LineDomain, RectDomain, CubeDomain)

    from dune.xt.grid import (Dim, Cube, Simplex, make_cube_grid, NormalBasedBoundaryInfo, DirichletBoundary,
            NeumannBoundary, RobinBoundary, UnknownBoundary)


    def discretize_domain_default(domain_description, diameter=1 / 100, grid_type=None):
        """Mesh a |DomainDescription| with dune-xt using an appropriate default implementation.

        This method can discretize the following |DomainDescriptions|:

            +----------------------+-----------+---------+
            | DomainDescription    | grid_type | default |
            +======================+===========+=========+
            | |LineDomain|         |      None |    X    |
            +----------------------+-----------+---------+
            |                      |   simplex |         |
            | |RectDomain|         +-----------+---------+
            |                      |      cube |    X    |
            +----------------------+-----------+---------+
            |                      |   simplex |         |
            | |CubeDomain|         +-----------+---------+
            |                      |      cube |    X    |
            +----------------------+-----------+---------+

        Parameters
        ----------
        domain_description
            A |DomainDescription| of the domain to mesh.
        diameter
            Maximal diameter of the codim-0 entities of the generated |Grid|.
        grid_type
            A string representing the type of codim-0 elements of the |Grid| which
            is to be constructed. If `None`, a default choice is made according to
            the table above.

        Returns
        -------
        grid
            The generated |Grid|.
        boundary_info
            The generated |BoundaryInfo|.
        """

        assert isinstance(diameter, Number)
        domain = domain_description.domain

        if isinstance(domain_description, LineDomain):

            if grid_type is not None:
                    raise NotImplementedError(f'I do not know how to create a {grid_type} grid!')

            grid = make_cube_grid(Dim(1), lower_left=[domain[0],], upper_right=[domain[1],],
                    num_elements=[int(m.ceil(1/diameter)),])

            bi = NormalBasedBoundaryInfo(grid, UnknownBoundary())
            for (bt, normal) in ((domain_description.left, [-1,]),
                                 (domain_description.right, [1,])):
                if bt == 'dirichlet':
                    bi.register_new_normal(normal, DirichletBoundary())
                elif bt == 'neumann':
                    bi.register_new_normal(normal, NeumannBoundary())
                elif bt == 'robin':
                    bi.register_new_normal(normal, RobinBoundary())
                else:
                    raise NotImplementedError(f'I do not know how to treat {bt} boundary type!')

            return grid, bi

        elif isinstance(domain_description, RectDomain):

            if grid_type == 'simplex':
                element_type = Simplex()
                num_elements = [int(m.ceil(domain_description.width / diameter)),
                                int(m.ceil(domain_description.height / diameter))]
            elif grid_type == 'cube' or grid_type is None:
                element_type = Cube()
                num_elements = [int(m.ceil(domain_description.width * m.sqrt(2) / diameter)),
                                int(m.ceil(domain_description.height * m.sqrt(2) / diameter))]
            else:
                raise NotImplementedError(f'I do not know how to create a {grid_type} grid!')

            grid = make_cube_grid(Dim(2), element_type,
                    lower_left=domain[0], upper_right=domain[1], num_elements=num_elements)

            bi = NormalBasedBoundaryInfo(grid, UnknownBoundary())
            for (bt, normal) in ((domain_description.left, [-1, 0]),
                                 (domain_description.right, [1, 0]),
                                 (domain_description.top, [0, 1]),
                                 (domain_description.bottom, [0, -1])):
                if bt == 'dirichlet':
                    bi.register_new_normal(normal, DirichletBoundary())
                elif bt == 'neumann':
                    bi.register_new_normal(normal, NeumannBoundary())
                elif bt == 'robin':
                    bi.register_new_normal(normal, RobinBoundary())
                else:
                    raise NotImplementedError(f'I do not know how to treat {bt} boundary type!')

            return grid, bi

        elif isinstance(domain_description, CubeDomain):

            if grid_type == 'simplex':
                element_type = Simplex()
                num_elements = [int(m.ceil(domain_description.width / diameter)),
                                int(m.ceil(domain_description.height / diameter)),
                                int(m.ceil(domain_description.depth / diameter))]
            elif grid_type == 'cube' or grid_type is None:
                element_type = Cube()
                num_elements = [int(m.ceil(domain_description.width * m.sqrt(3) / diameter)),
                                int(m.ceil(domain_description.height * m.sqrt(3) / diameter)),
                                int(m.ceil(domain_description.depth * m.sqrt(3) / diameter))]
            else:
                raise NotImplementedError(f'I do not know how to create a {grid_type} grid!')

            grid = make_cube_grid(Dim(3),
                    element_type, lower_left=domain[0], upper_right=domain[1], num_elements=num_elements)

            bi = NormalBasedBoundaryInfo(grid, UnknownBoundary())
            for (bt, normal) in ((domain_description.left, [-1, 0, 0]),
                                 (domain_description.right, [1, 0, 0]),
                                 (domain_description.top, [0, 1, 0]),
                                 (domain_description.bottom, [0, -1, 0]),
                                 (domain_description.front, [0, 0, -1]),
                                 (domain_description.back, [0, 0, 1])):
                if bt == 'dirichlet':
                    bi.register_new_normal(normal, DirichletBoundary())
                elif bt == 'neumann':
                    bi.register_new_normal(normal, NeumannBoundary())
                elif bt == 'robin':
                    bi.register_new_normal(normal, RobinBoundary())
                else:
                    raise NotImplementedError(f'I do not know how to treat {bt} boundary type!')

            return grid, bi

        else:
            raise NotImplementedError(f'I do not know how to discretize {domain_description} domain!')

