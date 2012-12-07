from __future__ import division
import math
import numpy as np
from matplotlib.delaunay import delaunay
import logging

from pymor.grid.interfaces import ISimpleAffineGrid, IConformalTopologicalGrid
from pymor.grid import referenceelements as Refs

class CircleBoundary(object):

    def __init__(self, radius=1, center=(0,0)):
        self.radius = radius
        self.center = center
        
    def __call__(self, p):
        '''for p in 0,1 returns the point on S1 in R2'''
        phi = 2 * math.pi * p
        return (self.center[0] + self.radius * math.sin(phi), self.center[1] + self.radius * math.cos(phi))

class StructuredSimplexCheat(object):
    def __init__(self, points, h):
        self.count = 0
        self.points = points
        self.rows = math.sqrt(points) 
        self.cols = self.rows
        self.h = h
        
    def __call__(self, p):
        x = (self.count % self.cols) * self.h
        y = (self.count // self.rows) * self.h
        self.count += 1
        if self.count >= self.points:
            self.count = 0
        return (x,y)
        
class PrescribedBoundaryGrid(IConformalTopologicalGrid):
    '''given an analytical boundary descriptjon f: [0,1)mapto R2
	I'll sample n points and construct a simplicial mesh of their convex hull
	'''
    dim = 2
    dim_outer = 2
    _ref_elements = {0: Refs.triangle, 1: Refs.line, 2: Refs.point} 

    def __init__(self, boundary_callable=CircleBoundary, sample_count=20):
        '''
    	:param boundary_callable: a callable we can sample sample_count times in 0,1
    	:param sample_count: how many points to sample from boundary_callable
    	if boundary function is of that fancy cas package we could sample 
    	weighed by the absolute value of the gradient. otherwise sampling is uniform
    	'''
        fac = 1 / sample_count
        self._px = np.array([boundary_callable(p * fac)[0] for p in range(sample_count)])
        self._py = np.array([boundary_callable(p * fac)[1] for p in range(sample_count)])
        self._cens, self._edg, self._tri, self._neig = delaunay(self._px, self._py)
        self._sizes = {0: len(self._tri), 1: len(self._edg), 2: sample_count}
    
    def subentities(self, codim, subentity_codim=None):
        if subentity_codim is None:
            subentity_codim = codim + 1
        if codim == 0:
            return np.array([(c[0], c[1], c[2]) for c in self._tri])
        else:
            return super(PrescribedBoundaryGrid, self).subentities(codim, subentity_codim)
            

    def reference_element(self, codim):
        return self._ref_elements[codim]
    
    def size(self, codim):
        return self._sizes[codim]
    
    def plot(self):
        import pylab
        for t in self.subentities(0):
            t = [t[i] for i in range(-1, 3)]
            pylab.plot(self._px[t], self._py[t])
            pylab.plot(self._px[t], self._py[t],'o')           
        pylab.show()
#        for t in self.subentities(1):
#            t = [t[i] for i in range(2)]
#            pylab.plot(self._px[t], self._py[t])           
#        pylab.show()
#        for t in self.subentities(2):
#            pylab.plot(self._px[t], self._py[t],'o')
#        pylab.show()
            
    def test_instances():
        return [PrescribedBoundaryGrid()]

if __name__ == "__main__":
    p = PrescribedBoundaryGrid(CircleBoundary(), sample_count=12)
    p.plot()
    c = 25
    p = PrescribedBoundaryGrid(StructuredSimplexCheat(c, 0.1), sample_count=c)
    p.plot()