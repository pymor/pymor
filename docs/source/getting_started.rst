.. _getting_started:

***************
Getting started
***************

Installation
------------

Before trying out pyMOR, you need to install it. We provide packages for Ubuntu
via our PPA::

        sudo apt-add-repository ppa:pymor/stable
        sudo apt-get update
        sudo apt-get install python-pymor python-pymor-demos python-pymor-doc

Daily snapshots can be installed by using the ``pymor/daily`` PPA instead of
``pymor/stable``. The current release can also be installed via `pip
<http://pip-installer.org>`. Please take a look at our `README
<https://github.com/pyMor/pyMor#installation-into-a-virtualenv>` file for
further details. The `README
<https://github.com/pyMor/pyMor#setting-up-an-environment-for-pymor-development>`
also contains instructions for setting up a development environment for working
on pyMOR itself.


Trying it out
-------------

While we consider pyMOR itself as a library for building MOR applications, we
ship a few example scripts with pyMOR itself. These can be found in the
``src/pymordemos`` directory of the source repository.  Try launching one of
them using the ``pymor-demo`` script provided by the ``python-pymor-demos``
package::

    pymor-demo thermalblock --with-estimator --plot-err --plot-solutions 3 2 3 32

The demo scripts can also be directly launched from the source tree::

    ./thermalblock.py --with-estimator --plot-err --plot-solutions 3 2 3 32

This will solve and reduce the so called thermalblock problem using
the reduced basis method with a greedy basis generation algorithm.
The thermalblock problem consists in solving the stationary diffusion
problem ::

    - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1     for x in Ω
                      u(x, μ)   = 0     for x in ∂Ω

on the domain Ω = [0,1]^2 for the unknown u. The domain is partitioned into
``XBLOCKS x YBLOCKS`` blocks (``XBLOCKS`` and ``YBLOCKS`` are the first 
two arguments to ``thermalblock.py``). The thermal conductivity d(x, μ)
is constant on each block (i,j) with value μ_ij: ::

    (0,1)------------------(1,1)
    |        |        |        |
    |  μ_11  |  μ_12  |  μ_13  |
    |        |        |        |
    |---------------------------
    |        |        |        |
    |  μ_21  |  μ_22  |  μ_23  |
    |        |        |        |
    (0,0)------------------(1,0)

The real numbers μ_ij form the ``XBLOCKS x YBLOCKS`` - dimensional parameter
on which the solution depends.

Running ``thermalblock.py`` will first produce plots of two detailed
solutions of the problem for different randomly chosen parameters
using linear finite elements. (The size of the grid can be controlled
via the ``--grid`` parameter. The randomly chosen parameters will
actually always be the same for each run, since a the random generator
is initialized with a fixed default seed in
:func:`~pymor.tools.random.new_random_state`.)

After closing the window, the reduced basis for model order reduction
is generated using a greedy search algorithm with error estimator.
The third parameter ``SNAPSHOTS`` of ``thermalblock.py`` determines how many
different values per parameter component μ_ij should be considered.
I.e. the parameter training set for basis generation will have the
size ``(XBLOCKS x YBLOCKS)^SNAPSHOTS``. After the basis of size 32 (the
last parameter) has been computed, the obtained reduced model (on the
32-dimensional reduced basis space) is evaluated by comparing the
solutions of the reduced and detailed models for new randomly chosen
parameters. Finally plots of the detailed and reduced solutions as well
as the difference between the two are displayed for the random parameter
for which a maximum reduction error has been measured.


The thermalblock demo explained
-------------------------------

In the following we will walk through the thermal block demo step by
step in an interactive Python shell. We assume that you are familiar
with the reduced basis method and that you know the basics of Python
programming as well as working with |NumPy|. (Note that our code will
differ a bit from ``thermalblock.py`` as we will hardcode the various
options the script offers and leave out some features.)

First, start a Python shell, we recommend using
`IPython <http://ipython.org>`_, which is also automatically installed
into the pyMOR virtualenv by the install script ::

    ipython

You can paste the following input lines starting with ``>>>`` by copying
them to the system clipboard and then executing ::

    %paste

inside the IPython shell.

To see what is going on, we will first adjust a few log levels of
pyMOR's logging facility:

>>> from pymor.core import set_log_levels
>>> set_log_levels({'pymor.algorithms': 'INFO',
...                 'pymor.discretizations': 'INFO'})
Loading pymor version 0.3.0

First we will instantiate a class describing the analytical problem
we want so solve. In this case, a 
:class:`~pymor.analyticalproblems.thermalblock.ThermalBlockProblem`:

>>> from pymor.analyticalproblems import ThermalBlockProblem
>>> p = ThermalBlockProblem(num_blocks=(3, 2))

Next we want to discretize this problem using the finite element method.
We could do this by hand, creating a |Grid|, instatiating
:class:`~pymor.operators.cg.DiffusionOperatorP1` finite element diffusion
operators for each subblock of the domain, forming a |LincombOperator|
by using :meth:`pymor.operators.interfaces.OperatorInterface.lincomb`
to represent the affine decomposition, instantiating a
:class:`~pymor.operators.cg.L2ProductFunctionalP1` as right hand side, and
putting it all together into a |StationaryDiscretization|. However, since
:class:`~pymor.analyticalproblems.thermalblock.ThermalBlockProblem` derives
form :class:`~pymor.analyticalproblems.elliptic.EllipticProblem`, we can use
a predifined *discretizer* to do the work for us. In this case, we use
:func:`~pymor.discretizers.elliptic.discretize_elliptic_cg`:

>>> import math as m
>>> from pymor.discretizers import discretize_elliptic_cg
>>> d, d_data = discretize_elliptic_cg(p, diameter=m.sqrt(2) / 100)

``d`` is the |StationaryDiscretization|, which has been created for us,
whereas ``d_data`` contains some additional data, in this case the |Grid|
and the |BoundaryInfo| which have been created during discretization. We
can have a look at the grid,

>>> print(d_data['grid'])
Tria-Grid on domain [0,1] x [0,1]
x0-intervals: 100, x1-intervals: 100
faces: 20000, edges: 30200, verticies: 10201

and as always, we can display its class documentation using
``help(d_data['grid'])``, or in the case of IPython
``d_data['grid']?``.

Let's solve the thermal block problem and visualize the solution:

>>> U = d.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
>>> d.visualize(U, title='Solution')
00:45|discretizations.basic.StationaryDiscretization: Solving ThermalBlock_CG for {diffusion: [1.0, 0.1, 0.3, 0.1, 0.2, 1.0]} ...

Each class in pyMOR that describes a |Parameter| dependent mathematical
object, like the |StationaryDiscretization| in our case, derives from
|Parametric| and determines the |Parameters| it expects during :meth:`__init__`
by calling :meth:`~pymor.parameters.base.Parametric.build_parameter_type`.
The resulting |ParameterType| is stored in the object's
:attr:`~pymor.parameters.base.Parametric.parameter_type` attribute. Let us
have a look:

>>> print(d.parameter_type)
{diffusion: (2, 3)}

This tells us, that the |Parameter| which
`~pymor.discretizations.interfaces.DiscretizationInterface.solve` expects
should be a dictionary with one key ``'diffusion'`` whose value is a
|NumPy array| of shape ``(2, 3)`` corresponding to the block structure of
the problem. However, by using the 
:meth:`~pymor.parameters.base.Parametric.parse_parameter` method, pyMOR is
smart enough to correctly parse the input ``[1.0, 0.1, 0.3, 0.1, 0.2, 1.0]``.

Next we want to use the :func:`~pymor.algorithms.greedy.greedy` algorithm
to reduce the problem. For this we need to choose a basis extension algorithm
as well as a reductor which will perform the actual RB-projection. We will
use :func:`~pymor.algorithms.basisextension.gram_schmidt_basis_extension` and
:func:`~pymor.reductors.linear.reduce_stationary_affine_linear`. The latter
will also assemble an error estimator to estimate the reduction error. This
will significantly speed up the basis generation, as we will only need to
solve the high-dimensional problem for those parameters in the training set
which are actually selected for basis extension. To control the condition of
the reduced system matrix, we must ensure that the generated basis is
orthonormal w.r.t. the H1-product on the solution space. For this we provide
the basis extension algorithm with the :attr:`h1_product` attribute of the
discretization.

>>> from functools import partial
>>> from pymor.algorithms.greedy import greedy
>>> from pymor.algorithms.basisextension import gram_schmidt_basis_extension
>>> from pymor.reductors.linear import reduce_stationary_affine_linear
>>> extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_product)

Moreover, we need to select a |Parameter| training set. The discretization
``d`` already comes with a |ParameterSpace| it has obtained from the analytical
problem. We can sample our parameters from this space, which is a
:class:`~pymor.parameters.spaces.CubicParameterSpace`. E.g.:

>>> samples = list(d.parameter_space.sample_uniformly(2))
>>> print(samples[0])
{diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

Now we start the basis generation:

>>> greedy_data = greedy(d, reduce_stationary_affine_linear, samples,
...                      extension_algorithm=extension_algorithm,
...                      use_estimator=True, max_extensions=32)
01:32|algorithms.greedy.greedy: Started greedy search on 64 samples                                                                                   
01:32|algorithms.greedy.greedy: Reducing ...                                                                                                          
01:32|algorithms.greedy.greedy: Estimating errors ...                                                                                                 
01:32|algorithms.greedy.greedy: Maximum error after 0 extensions: 0.0099 (mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})                           
01:32|algorithms.greedy.greedy: Extending with snapshot for mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}                                          
01:32|discretizations.basic.StationaryDiscretization: Solving ThermalBlock_CG for {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]} ...
    ...
    ...
01:50|algorithms.greedy.greedy: Maximal number of 32 extensions reached.
01:50|algorithms.greedy.greedy: Reducing once more ...
01:50|algorithms.greedy.greedy: Greedy search took 17.7437560558 seconds

The ``max_extensions`` parameter defines how many basis vectors we want to
obtain. ``greedy_data`` is a dictionary containing various data that has
been generated during the run of the algorithm:

>>> print(greedy_data.keys())
['time', 'reduction_data', 'reconstructor', 'max_err', 'max_err_mus', 'basis', 'extensions', 'reduced_discretization', 'max_err_mu', 'max_errs']

The most important items are ``'reduced_discretization'`` and
``'reconstructor'``, which hold the reduced |Discretization| obtained
from applying our reductor with the final reduced basis, as well as a
reconstructor to reconstruct detailed solutions from the reduced solution
vectors. The reduced basis is stored in the ``'basis'`` item.

>>> rd = greedy_data['reduced_discretization']
>>> rc = greedy_data['reconstructor']
>>> rb = greedy_data['basis']

All vectors in pyMOR are stored in so called |VectorArrays|. For example
the solution ``U`` computed above is given as a |VectorArray| of length 1.
For the reduced basis we have:

>>> print(type(rb))
<class 'pymor.la.numpyvectorarray.NumpyVectorArray'>
>>> print(len(rb))
32
>>> print(rb.dim)
10201

Let us check, if the reduced basis really is orthonormal with respect to
the H1-product. For this we use the :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
method:

>>> import numpy as np
>>> gram_matrix = d.h1_product.apply2(rb, rb, pairwise=False)
>>> print(np.max(np.abs(gram_matrix - np.eye(32))))
2.17350009518e-15

Looks good! We can now solve the reduced model for the same parameter as above.
The result is a vector of coefficients w.r.t. the reduced basis, which is
currently stored in ``rb``. To form the linear combination, we use the
reconstructor:

>>> u = rd.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
>>> print(u)
[[  5.65450212e-01  -9.97259318e-03  -1.37904584e-01   1.49072806e-01
    1.38146480e-01   8.32847282e-02  -2.36482451e-01   1.01121628e-01
    1.03270816e-01  -3.18681618e-02   4.17663255e-02   2.92689535e-02
    9.12690185e-02  -7.58645640e-02   1.36683727e-01   9.88630906e-02
   -9.66481730e-03  -3.74264667e-03  -1.80396304e-03   8.29032084e-03
   -1.66055113e-02   1.27241150e-02   1.42330922e-02   8.98507806e-03
    6.31953865e-03   7.52031711e-04   1.35377961e-03   3.77849546e-03
    1.27019758e-03   3.75581650e-03   7.22952797e-04   5.64761035e-04]]
>>> U_red = rc.reconstruct(u)
>>> print(U_red.dim)
10201

Finally we compute the reduction error and display the reduced solution along with
the detailed solution and the error:

>>> ERR = U - U_red
>>> print(d.h1_norm(ERR))
[ 0.00307307]
>>> d.visualize((U, U_red, ERR), legend=('Detailed', 'Reduced', 'Error'),
...             separate_colorbars=True)

We can nicely observe how the error is maximized along the jumps of the
diffusion coeffient, which is expected.

Learning more
-------------

As a next step, you should read our :ref:`technical_overview` which discusses the
most important concepts and design decisions behind pyMOR. After that
you should be fit to delve into the reference documentation.

Should you have any problems regarding pyMOR, questions or
`feature requests <https://github.com/pymor/pymor/issues>`_, do not hestitate
to contact us at our
`mailing list <http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev>`_!
