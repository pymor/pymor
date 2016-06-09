# pyMOR main developers

* René Milk, rene.milk@uni-muenster.de
* Stephan Rave, stephan.rave@uni-muenster.de
* Felix Schindler, felix.schindler@uni-muenster.de


# Contributors

## pyMOR 0.5

* Petar Mlinarić, mlinaric@mpi-magdeburg.mpg.de
    * l2_norm2 methods for VectorArrays
    * from_data methods for VectorSpace and VectorArrays
    * algorithms.to_matrix


## pyMOR 0.4

* Andreas Buhr, andreas@andreasbuhr.de
    * ability to rescale colorbar in each frame
    * SelectionOperator
    * support for advection and rection terms in finite element discretizations
    * improved Robin boundary condition support

* Michael Laier, m_laie01@uni-muenster.de
    * PolygonalDomain, CircularSectorDomain, DiscDomain
    * pymor.domaindiscretizers.gmsh
    * ParabolicProblem, discretize_parabolic_cg, discretize_parabolic_fv
    * reductors.parabolic
    * reductors.residual.reduce_implicit_euler_residual
    * pymordemos.parabolic, pymordemos.parabolic_mor
    * ProductParameterFunctional

* Falk Meyer, falk.meyer@uni-muenster.de
    * pymor.discretizers.disk

* Petar Mlinarić, mlinaric@mpi-magdeburg.mpg.de
    * complex number support for NumpyVectorArray and NumpyMatrixOperator
    * BlockOperator and BlockDiagonalOperator

* Michael Schaefer, michael.schaefer@uni-muenster.de
    * Robin bondary condition support for pymor.operators.cg


## pyMOR 0.3

* Andreas Buhr, andreas@andreasbuhr.de
    * improved PIL compatibility for BitmapFunction
    * improvements to Gram-Schmidt algorithm

* Lucas Camphausen, lucascamp@web.de
    * bilinear finite elements

* Michael Laier, m_laie01@uni-muenster.de
    * finite volume diffusion operator

* Michael Schaefer, michael.schaefer@uni-muenster.de
    * new 'columns' parameter for PatchVisualizer


## pyMOR 0.2

* Andreas Buhr, andreas@andreasbuhr.de
    * reiteration procedure in Gram-Schmidt algorithm for improved numerical
      stability

* Michael Laier, m_laie01@uni-muenster.de
    * documentation improvements


# Detailed information

Detailed contribution information can be obtained from the revision history
of pyMOR's [git repository](https://github.com/pymor/pymor/graphs/contributors?type=c).
