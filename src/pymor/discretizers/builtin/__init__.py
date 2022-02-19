# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.discretizers.builtin.cg import discretize_stationary_cg, discretize_instationary_cg
from pymor.discretizers.builtin.fv import discretize_stationary_fv, discretize_instationary_fv
from pymor.discretizers.builtin.grids.gmsh import load_gmsh
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.grids.tria import TriaGrid
