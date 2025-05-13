# pyMOR developers and contributors

## Main developers

* Linus Balicki, @lbalicki (2020-)
* René Fritze, @renefritze (2012-2024)
* Hendrik Kleikamp, @HenKlei (2022-)
* Petar Mlinarić, @pmli (2016-)
* Stephan Rave, @sdrave (2012-)
* Felix Schindler, @ftschindler (2012-)

## Contributors

### pyMOR 2025.1

* Timo Plath @TiPlath
  * Adjusted neural network reductor to run without full order model

### pyMOR 2024.2

* Maximilian Bindhak, @maxbindhak
  * Cholesky QR with recalculated shifts
  * `return_R` parameter for `shifted_chol_qr`

* Dmitry Kabanov, @dmitry-kabanov
  * use correct Python executable when running `pymor_vis` script

* Art Pelling, @artpelling
  * leave-one-out error estimator for adaptive randomized range approximation
  * support for `sparray` in `from_matrices` constructors in `pymor.models.iosys`

### pyMOR 2024.1

* Maximilian Bindhak, @maxbindhak
  * fixes in shifted Cholesky QR algorithm
  * performance improvements in `NumpyVectorArray` implementation

* Art Pelling, @artpelling
  * various improvements in `rand_la` module
  * shifted Cholesky QR algorithm

### pyMOR 2023.2

* Steffen Müller, @steff-mueller
  * tutorial for port-Hamiltonian systems
  * interface for dense solvers of positive Riccati equations and dense positive-real Gramians

* Peter Oehme, @peoe
  * self-documenting help target for Makefile
  * Operator-valued functions in FactorizedTransferFunction
  * H2 norm preset for TransferFunction

* Art Pelling, @artpelling
  * matrix-free implementation of circulant, Hankel and Toeplitz operators
  * computation of an LTIModel's forced response

### pyMOR 2023.1

* Tim Keil, @TiKeil
  * adaptive trust region algorithm and specific BFGS implementation for PDE-constrained optimization

* Steffen Müller, @steff-mueller
  * positive-real balanced truncation
  * passivity preserving model reduction via spectral factorization

* Mohamed Adel Naguib Ahmed, @MohamedAdelNaguib
  * input-output selection in `bode_plot` function

* Jonas Nicodemus, @Jonas-Nicodemus
  * port-Hamiltonian IRKA
  * positive-real balanced truncation

* Peter Oehme, @peoe
  * quadratic functionals and quadratic output keyword for CG discretization
  * simple algebraic operations for parameter values
  * adaptive trust region algorithm and specific BFGS implementation for PDE-constrained optimization

### pyMOR 2022.2

* Tim Keil, @TiKeil
  * Dual-weighted residual (DWR) output estimation for elliptic problems

* Art Pelling, @artpelling
  * Eigensystem Realization Algorithm

### pyMOR 2022.1

* Patrick Buchfink, @pbuchfink
  * symplectic model order reduction

* Monica Dessole, @mdessole
  * Navier-Stokes demo using neural networks

* Hendrik Kleikamp, @HenKlei
  * several additional features for neural networks
  * purely data-driven usage of neural networks without requiring full-order model
  * Navier-Stokes demo using neural networks

* Peter Oehme, @peoe
  * support for UFL expression conversion

* Art Pelling, @artpelling
  * functionality to instantiate non-parametric LTIModels with preset attributes
  * support for discrete-time LTI systems, Lyapunov equations and balanced truncation
  * Moebius transformations and continuous/discrete-time conversion of LTI models
  * dedicated Hankel operator class

* Sven Ullmann, @ullmannsven
  * randomized algorithms for generalized SVD and generalized Hermitian eigenvalue problem

### pyMOR 2021.2

* Tim Keil, @TiKeil
  * Simple output estimation for elliptic and parabolic problems

* Jonas Nicodemus, @Jonas-Nicodemus
  * dynamic mode decomposition

* Henrike von Hülsen, h.vonhuelsen@uni-muenster.de
  * dynamic mode decomposition

### pyMOR 2021.1

* Meret B., @meretp
  * modal truncation for model order reduction

* Hendrik Kleikamp, @HenKlei
  * artificial neural networks for approximation of output quantities

### pyMOR 2020.2

* Tim Keil, @TiKeil
  * energy product in elliptic discretizer
  * rename estimate --> estimate_error and estimator -> error_estimator
  * avoid nested Product and Lincomb Functionals and Functions
  * linear optimization (dual solution, sensitivities, output gradient)

* Hendrik Kleikamp, @HenKlei
  * artificial neural networks for instationary problems

### pyMOR 2020.1

* Linus Balicki, @lbalicki
  * implicitly restarted Arnoldi method for eigenvalue computation in
    algorithms.eigs
  * subspace accelerated dominant pole algorithm in algorithms.samdp

* Tim Keil, @TiKeil
  * second order derivatives for parameters
  * speed up of LincombOperators
  * add LincombParameterFunctional
  * product rule for ProductParameterFunctional
  * BaseMaxThetaParameterFunctional

* Hendrik Kleikamp, @HenKlei
  * Armijo line search algorithm
  * artificial neural networks for model order reduction

* Luca Mechelli, @mechiluca
  * speed up of LincombOperators

### pyMOR 2019.2

* Linus Balicki, @lbalicki
  * low-rank RADI method for Riccati equations in algorithms.lrradi
  * improve projection shifts for low-rank ADI method for Lyapunov equations

* Dennis Eickhorn, @deneick
  * randomized range approximation algorithms in algorithms.randrangefinder
  * fixed complex norms in vectorarrays.interfaces

* Tim Keil, @TiKeil
  * partial derivatives for parameters d_mu
  * affine decomposition of robin operator and rhs functionals

### pyMOR 0.5

* Linus Balicki, @lbalicki
  * low-rank ADI method using projection shifts for Lyapunov equations in
    algorithms.lyapunov

* Julia Brunken, @JuliaBru
  * support for advection, reaction terms and Robin boundary data in
    ParabolicProblem

* Christoph Lehrenfeld, @schruste
  * contributions to NGSolve wrappers
  * NGSolve model in thermalblock_simple.py

### pyMOR 0.4

* Andreas Buhr, @andreasbuhr
  * ability to rescale colorbar in each frame
  * SelectionOperator
  * support for advection and reaction terms in finite element discretizations
  * improved Robin boundary condition support

* Michael Laier, @michaellaier
  * PolygonalDomain, CircularSectorDomain, DiscDomain
  * pymor.domaindiscretizers.gmsh
  * ParabolicProblem, discretize_parabolic_cg, discretize_parabolic_fv
  * reductors.parabolic
  * reductors.residual.reduce_implicit_euler_residual
  * pymordemos.parabolic, pymordemos.parabolic_mor
  * ProductParameterFunctional

* Falk Meyer, falk.meyer@uni-muenster.de
  * pymor.discretizers.disk

* Petar Mlinarić, @pmli
  * complex number support for NumpyVectorArray and NumpyMatrixOperator
  * BlockOperator and BlockDiagonalOperator

* Michael Schaefer, @michaelschaefer
  * Robin boundary condition support for pymor.operators.cg

### pyMOR 0.3

* Andreas Buhr, @andreasbuhr
  * improved PIL compatibility for BitmapFunction
  * improvements to Gram-Schmidt algorithm

* Lucas Camphausen, lucascamp@web.de
  * bilinear finite elements

* Michael Laier, @michaellaier
  * finite volume diffusion operator

* Michael Schaefer, @michaelschaefer
  * new 'columns' parameter for PatchVisualizer

### pyMOR 0.2

* Andreas Buhr, @andreasbuhr
  * reiteration procedure in Gram-Schmidt algorithm for improved numerical
    stability

* Michael Laier, @michaellaier
  * documentation improvements

## Detailed information

Detailed contribution information can be obtained from the revision history
of pyMOR's [git repository](https://github.com/pymor/pymor/graphs/contributors?type=c).
