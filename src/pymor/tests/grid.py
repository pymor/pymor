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
import unittest

def make_testcase_classes(grid_types, TestCase):
    for GridType in grid_types:
        if GridType.has_interface_name():
            continue
        cname = '{}Test'.format(GridType.__name__)
        #saves a new type called cname with correct bases and class dict in globals
        globals()[cname] = type(cname, (unittest.TestCase, TestCase), {'grid': GridType()})


class SimpleAffineGridTest(object):
    
    def test_volumes(self):
        grid = self.grid
        dim = grid.dim
        for codim in range(dim+1):
            self.assertGreater(np.argmin(grid.volumes(codim)), 0)
            self.assertGreater(np.argmin(grid.volumes_inverse(codim)), 0)
            self.assertGreater(np.argmin(grid.diameters(codim)), 0)    

class ConformalTopologicalGridTest(object):
    
    def test_sizes(self):
        grid = self.grid
        dim = grid.dim
        for codim in range(dim+1):
            self.assertGreater(grid.size(codim), 0)

make_testcase_classes(ISimpleAffineGrid.implementors(True), SimpleAffineGridTest)
#this hard fails module import atm
#make_testcase_classes(IConformalTopologicalGrid.implementors(True), ConformalTopologicalGridTest)
        
if __name__ == "__main__":
#    nose.core.runmodule(name='__main__')
    logging.basicConfig(level=logging.INFO)
    unittest.main()