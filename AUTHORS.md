# pyMOR developers and contributors

## Main developers

* Linus Balicki, balicki@vt.edu (2020-)
* René Fritze, rene.fritze@uni-muenster.de (2012-)
* Petar Mlinarić, mlinaric@vt.edu (2016-)
* Stephan Rave, stephan.rave@uni-muenster.de (2012-)
* Felix Schindler, felix.schindler@uni-muenster.de (2012-)

## Contributors

### pyMOR 2021.2

* Hendrik Kleikamp, hendrik.kleikamp@uni-muenster.de
  * long short-term memory neural networks for instationary problems

### pyMOR 2021.1

* Meret Behrens, mbehrens@mpi-magdeburg.mpg.de
  * modal truncation for model order reduction

* Hendrik Kleikamp, hendrik.kleikamp@uni-muenster.de
  * artificial neural networks for approximation of output quantities

### pyMOR 2020.2

* Tim Keil, tim.keil@uni-muenster.de
  * energy product in elliptic discretizer
  * rename estimate --> estimate_error and estimator -> error_estimator
  * avoid nested Product and Lincomb Functionals and Functions
  * linear optimization (dual solution, sensitivities, output gradient)

* Hendrik Kleikamp, hendrik.kleikamp@uni-muenster.de
  * artificial neural networks for instationary problems

### pyMOR 2020.1

* Linus Balicki, linus.balicki@ovgu.de
  * implicitly restarted Arnoldi method for eigenvalue computation in
    algorithms.eigs
  * subspace accelerated dominant pole algorithm in algorithms.samdp

* Tim Keil, tim.keil@uni-muenster.de
  * second order derivatives for parameters
  * speed up of LincombOperators
  * add LincombParameterFunctional
  * pruduct rule for ProductParameterFunctional
  * BaseMaxThetaParameterFunctional

* Hendrik Kleikamp, hendrik.kleikamp@uni-muenster.de
  * Armijo line search algorithm
  * artificial neural networks for model order reduction

* Luca Mechelli, luca.mechelli@uni-konstanz.de
  * speed up of LincombOperators

### pyMOR 2019.2

* Linus Balicki, linus.balicki@ovgu.de
  * low-rank RADI method for Riccati equations in algorithms.lrradi
  * improve projection shifts for low-rank ADI method for Lyapunov equations

* Dennis Eickhorn, d.eickhorn@uni-muenster.de
  * randomized range approximation algorithms in algorithms.randrangefinder
  * fixed complex norms in vectorarrays.interfaces

* Tim Keil, tim.keil@uni-muenster.de
  * partial derivatives for parameters d_mu
  * affine decomposition of robin operator and rhs functionals

### pyMOR 0.5

* Linus Balicki, linus.balicki@ovgu.de
  * low-rank ADI method using projection shifts for Lyapunov equations in
    algorithms.lyapunov

* Julia Brunken, julia.brunken@uni-muenster.de
  * support for advection, reaction terms and Robin boundary data in
    ParabolicProblem

* Christoph Lehrenfeld, lehrenfeld@math.uni-goettingen.de
  * contributions to NGSolve wrappers
  * NGSolve model in thermalblock_simple.py

### pyMOR 0.4

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

### pyMOR 0.3

* Andreas Buhr, andreas@andreasbuhr.de
  * improved PIL compatibility for BitmapFunction
  * improvements to Gram-Schmidt algorithm

* Lucas Camphausen, lucascamp@web.de
  * bilinear finite elements

* Michael Laier, m_laie01@uni-muenster.de
  * finite volume diffusion operator

* Michael Schaefer, michael.schaefer@uni-muenster.de
  * new 'columns' parameter for PatchVisualizer

### pyMOR 0.2

* Andreas Buhr, andreas@andreasbuhr.de
  * reiteration procedure in Gram-Schmidt algorithm for improved numerical
    stability

* Michael Laier, m_laie01@uni-muenster.de
  * documentation improvements

## Detailed information

Detailed contribution information can be obtained from the revision history
of pyMOR's [git repository](https://github.com/pymor/pymor/graphs/contributors?type=c).
