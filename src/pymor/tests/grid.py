'''
Created on Nov 16, 2012

@author: r_milk01
'''
import numpy as np
import logging
import nose
import pprint
from pymor.grid.interfaces import IConformalTopologicalGrid, ISimpleAffineGrid 
#mandatory so all Grid classes are created
from pymor.grid import *


def check_sizes(grid):
    dim = grid.dim
    for codim in range(dim+1):
        size = grid.size(codim)
        assert size > 0, grid

def check_volumes(grid):
    dim = grid.dim
    for codim in range(dim+1):
        assert np.argmin(grid.volumes(codim)) > 0, grid
        assert np.argmin(grid.volumes_inverse(codim)) > 0, grid
        assert np.argmin(grid.diameters(codim)) > 0, grid

def test_interface_conformance():
    c_grids = set([g() for g in IConformalTopologicalGrid.implementors(True) if not g.has_interface_name()])
    s_grids = set([g() for g in ISimpleAffineGrid.implementors(True) if not g.has_interface_name()])
    logging.error('Testing IConformalTopologicalGrid implementors: %s', 
                  pprint.pformat([c.__class__.__name__ for c in c_grids]))
    logging.error('Testing ISimpleAffineGrid implementors: %s', 
                  pprint.pformat([s.__class__.__name__ for s in s_grids]))
    for g in c_grids:
        yield check_sizes, g
    for g in s_grids:
        yield check_volumes, g
        
if __name__ == "__main__":
    nose.core.runmodule(name='__main__')