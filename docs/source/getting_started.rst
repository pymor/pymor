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
``pymor/stable``. The current release can also be installed via
`pip <http://pip-installer.org>`_. Please take a look at our
`README <https://github.com/pymor/pymor#installation-into-a-virtualenv>`_ file for
further details. The
`README <https://github.com/pymor/pymor#setting-up-an-environment-for-pymor-development>`_
also contains instructions for setting up a development environment for working
on pyMOR itself.


Trying it out
-------------

While we consider pyMOR mainly as a library for building MOR applications, we
ship a few example scripts. These can be found in the ``src/pymordemos``
directory of the source repository.  Try launching one of
them using the ``pymor-demo`` script contained in the ``python-pymor-demos``
package::

    pymor-demo thermalblock --plot-err --plot-solutions 3 2 3 32

The demo scripts can also be launched directly from the source tree::

    ./thermalblock.py --plot-err --plot-solutions 3 2 3 32

This will solve and reduce the so called thermal block problem using
the reduced basis method with a greedy basis generation algorithm.
The thermal block problem consists in solving the stationary diffusion
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
actually be the same for each run, since a the random generator
is initialized with a fixed default seed in
:func:`~pymor.tools.random.new_random_state`.)

After closing the window, the reduced basis for model order reduction
is generated using a greedy search algorithm with error estimator.
The third parameter ``SNAPSHOTS`` of ``thermalblock.py`` determines how many
different values per parameter component μ_ij should be considered.
I.e. the parameter training set for basis generation will have the
size ``SNAPSHOTS^(XBLOCKS x YBLOCKS)``. After the basis of size 32 (the
last parameter) has been computed, the quality obtained reduced model
(on the 32-dimensional reduced basis space) is evaluated by comparing the
solutions of the reduced and detailed models for new randomly chosen
parameters. Finally plots of the detailed and reduced solutions as well
as the difference between the two are displayed for the random parameter
which maximises reduction error.


The thermalblock demo explained
-------------------------------

In the following we will walk through the thermal block demo step by
step in an interactive Python shell. We assume that you are familiar
with the reduced basis method and that you know the basics of
`Python <http://www.python.org>`_ programming as well as working
with |NumPy|. (Note that our code will differ a bit from
``thermalblock.py`` as we will hardcode the various options the script
offers and leave out some features.)

First, start a Python shell. We recommend using
`IPython <http://ipython.org>`_ ::

    ipython

You can paste the following input lines starting with ``>>>`` by copying
them to the system clipboard and then executing ::

    %paste

inside the IPython shell.

First, we will import the most commonly used methods and classes of pyMOR
by executing:

>>> from pymor.basic import *
Loading pymor version 0.3.0

Next we will instantiate a class describing the analytical problem
we want so solve. In this case, a 
:class:`~pymor.analyticalproblems.thermalblock.ThermalBlockProblem`:

>>> p = ThermalBlockProblem(num_blocks=(3, 2))

We want to discretize this problem using the finite element method.
We could do this by hand, creating a |Grid|, instatiating
:class:`~pymor.operators.cg.DiffusionOperatorP1` finite element diffusion
operators for each subblock of the domain, forming a |LincombOperator|
to represent the affine decomposition, instantiating a
:class:`~pymor.operators.cg.L2ProductFunctionalP1` as right hand side, and
putting it all together into a |StationaryDiscretization|. However, since
:class:`~pymor.analyticalproblems.thermalblock.ThermalBlockProblem` derives
form :class:`~pymor.analyticalproblems.elliptic.EllipticProblem`, we can use
a predifined *discretizer* to do the work for us. In this case, we use
:func:`~pymor.discretizers.elliptic.discretize_elliptic_cg`:

>>> d, d_data = discretize_elliptic_cg(p, diameter=1. / 100.)

``d`` is the |StationaryDiscretization| which has been created for us,
whereas ``d_data`` contains some additional data, in this case the |Grid|
and the |BoundaryInfo| which have been created during discretization. We
can have a look at the grid,

>>> print(d_data['grid'])
Tria-Grid on domain [0,1] x [0,1]
x0-intervals: 100, x1-intervals: 100
faces: 40000, edges: 60200, vertices: 20201

and, as always, we can display its class documentation using
``help(d_data['grid'])``, or in the case of IPython
``d_data['grid']?``.

Let's solve the thermal block problem and visualize the solution:

>>> U = d.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
>>> d.visualize(U, title='Solution')
00:45|discretizations.basic.StationaryDiscretization: Solving ThermalBlock_CG for {diffusion: [1.0, 0.1, 0.3, 0.1, 0.2, 1.0]} ...
    ...
    ...

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
:func:`~pymor.reductors.stationary.reduce_stationary_coercive`. The latter
will also assemble an error estimator to estimate the reduction error. This
will significantly speed up the basis generation, as we will only need to
solve the high-dimensional problem for those parameters in the training set
which are actually selected for basis extension. To control the condition of
the reduced system matrix, we must ensure that the generated basis is
orthonormal w.r.t. the H1-product on the solution space. For this we pass
the basis extension algorithm the :attr:`h1_product` attribute of the
discretization. We pass the same product to the reductor for computing the
Riesz representatives for error estimation. Moreover, we have to provide
a |ParameterFunctional| which computes a lower bound for the coercivity of
the problem for a given parameter.

>>> from functools import partial
>>> extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_product)
>>> reductor = partial(reduce_stationary_coercive, error_product=d.h1_product,
                       coercivity_estimator=GenericParameterFunctional(lambda mu: np.min(mu['diffusion']),
                                                                       d.parameter_type))

Moreover, we need to select a |Parameter| training set. The discretization
``d`` already comes with a |ParameterSpace| which it has inherited from the
analytical problem. We can sample our parameters from this space, which is a
:class:`~pymor.parameters.spaces.CubicParameterSpace`. E.g.:

>>> samples = list(d.parameter_space.sample_uniformly(4))
>>> print(samples[0])
{diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

Now we start the basis generation:

>>> greedy_data = greedy(d, reductor, samples,
...                      extension_algorithm=extension_algorithm,
...                      use_estimator=True, max_extensions=32)
07:42|algorithms.greedy.greedy: Started greedy search on 4096 samples
07:42|algorithms.greedy.greedy: Reducing ...
07:42|algorithms.greedy.greedy: Estimating errors ...
    ...
07:44|algorithms.greedy.greedy: Maximum error after 0 extensions: 9.86736953629 (mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})
07:44|algorithms.greedy.greedy: Extending with snapshot for mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}
07:44|discretizations.basic.StationaryDiscretization: Solving ThermalBlock_CG for {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]} ...
    ...
    ...
15:26|algorithms.greedy.greedy: Maximum number of 32 extensions reached.
15:26|algorithms.greedy.greedy: Reducing once more ...
15:55|algorithms.greedy.greedy: Greedy search took 492.942929029 seconds

The ``max_extensions`` parameter defines how many basis vectors we want to
obtain. ``greedy_data`` is a dictionary containing various data that has
been generated during the run of the algorithm:

>>> print(greedy_data.keys())
['reduction_data', 'reconstructor', 'time', 'basis', 'extensions', 'reduced_discretization', 'max_errs', 'max_err_mus']

The most important items are ``'reduced_discretization'`` and
``'reconstructor'`` which hold the reduced |Discretization| obtained
from applying our reductor with the final reduced basis, as well as a
reconstructor to reconstruct detailed solutions from the reduced solution
vectors. The reduced basis is stored as ``'basis'`` item.

>>> rd = greedy_data['reduced_discretization']
>>> rc = greedy_data['reconstructor']
>>> rb = greedy_data['basis']

All vectors in pyMOR are stored in so called |VectorArrays|. For example
the solution ``U`` computed above is given as a |VectorArray| of length 1.
For the reduced basis we have:

>>> print(type(rb))
<class 'pymor.vectorarrays.numpy.NumpyVectorArray'>
>>> print(len(rb))
32
>>> print(rb.dim)
20201

Let us check if the reduced basis really is orthonormal with respect to
the H1-product. For this we use the :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
method:

>>> import numpy as np
>>> gram_matrix = d.h1_product.apply2(rb, rb)
>>> print(np.max(np.abs(gram_matrix - np.eye(32))))
1.24982272795e-13

Looks good! We can now solve the reduced model for the same parameter as above.
The result is a vector of coefficients w.r.t. the reduced basis, which is
currently stored in ``rb``. To form the linear combination, we can use the
reconstructor:

>>> u = rd.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
>>> print(u)
[[  5.79477471e-01   5.91289054e-02   1.89924036e-01   1.89149529e-02
    1.81103127e-01   2.69920752e-02  -1.79611519e-01   7.99676272e-03
    1.54092560e-01   5.76326362e-02   1.97982347e-01  -2.13775254e-02
    3.12892660e-02  -1.27037440e-01  -1.51352508e-02   3.36101087e-02
    2.05779889e-02  -4.96445984e-03   3.21176662e-02  -2.52674851e-02
    2.92150040e-02   3.23570362e-03  -4.14288199e-03   5.48325425e-03
    4.10728945e-03   1.59251955e-03  -9.23470903e-03  -2.57483574e-03
   -2.52451212e-03  -5.08125873e-04   2.71427033e-03   5.83210112e-05]]
>>> U_red = rc.reconstruct(u)
>>> print(U_red.dim)
20201

Finally we compute the reduction error and display the reduced solution along with
the detailed solution and the error:

>>> ERR = U - U_red
>>> print(d.h1_norm(ERR))
[ 0.00944595]
>>> d.visualize((U, U_red, ERR), legend=('Detailed', 'Reduced', 'Error'),
...             separate_colorbars=True)

We can nicely observe that, as expected, the error is maximized along the
jumps of the diffusion coeffient.

Learning more
-------------

As a next step, you should read our :ref:`technical_overview` which discusses the
most important concepts and design decisions behind pyMOR. After that
you should be fit to delve into the reference documentation.

Should you have any problems regarding pyMOR, questions or
`feature requests <https://github.com/pymor/pymor/issues>`_, do not hestitate
to contact us at our
`mailing list <http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev>`_!
