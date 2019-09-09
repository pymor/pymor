.. _getting_started:

***************
Getting started
***************

Installation
------------

pyMOR can be easily installed via `pip`::

        pip install --upgrade pip  # make sure that pip is reasonably new
        pip install pymor[full]

For Linux we provide binary wheels, so no further system packages should
be required. The following optional packages must be installed separately::

        # for support of MPI distributed models and parallelization of
        # greedy algorithms (requires MPI development headers and a C compiler)
        pip install mpi4py

        # dense matrix equation solver for system-theoretic MOR methods,
        # required for H-infinity norm calculation (requires OpenBLAS headers and a Fortran compiler)
        pip install slycot

        # sparse matrix equation solver for system-theoretic MOR methods
        # (other backends available)
        open https://www.mpi-magdeburg.mpg.de/projects/mess
        # download and install pymess wheel for your architecture

We recommend installation of pyMOR in a `virtual environment <https://virtualenv.pypa.io/en/latest/>`_.
Please take a look at our `README <https://github.com/pymor/pymor#installation-via-pip>`_
file for more detailed installation instructions and a guide to setup a
development environment for working on pyMOR itself.


Trying it out
-------------

While we consider pyMOR mainly as a library for building MOR applications, we
ship a few example scripts. These can be found in the ``src/pymordemos``
directory of the source repository (some are available as Jupyter notebooks in
the ``notebooks`` directory). Try launching one of them using the ``pymor-demo``
script::

    pymor-demo thermalblock --plot-err --plot-solutions 3 2 3 32

The demo scripts can also be launched directly from the source tree::

    ./thermalblock.py --plot-err --plot-solutions 3 2 3 32

This will reduce the so called thermal block problem using the reduced basis
method with a greedy basis generation algorithm. The thermal block problem
consists in solving the stationary heat equation ::

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
:func:`~pymor.tools.random.default_random_state`.)

After closing the window, the reduced basis for model order reduction
is generated using a greedy search algorithm with error estimator.
The third parameter ``SNAPSHOTS`` of ``thermalblock.py`` determines how many
different values per parameter component μ_ij should be considered.
I.e. the parameter training set for basis generation will have the
size ``SNAPSHOTS^(XBLOCKS x YBLOCKS)``. After the basis of size 32 (the
last parameter) has been computed, the quality of the obtained reduced model
(on the 32-dimensional reduced basis space) is evaluated by comparing the
solutions of the reduced and detailed models for new, randomly chosen
parameters. Finally, plots of the detailed and reduced solutions, as well
as the difference between the two, are displayed for the random parameter
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

Next we will instantiate a class describing the analytical problem
we want so solve. In this case, a
:meth:`~pymor.analyticalproblems.thermalblock.thermal_block_problem`:

>>> p = thermal_block_problem(num_blocks=(3, 2))

We want to discretize this problem using the finite element method.
We could do this by hand, creating a |Grid|, instatiating
:class:`~pymor.operators.cg.DiffusionOperatorP1` finite element diffusion
operators for each subblock of the domain, forming a |LincombOperator|
to represent the affine decomposition, instantiating a
:class:`~pymor.operators.cg.L2ProductFunctionalP1` as right hand side, and
putting it all together into a |StationaryModel|. However, since
:meth:`~pymor.analyticalproblems.thermalblock.thermal_block_problem` returns
a :class:`~pymor.analyticalproblems.elliptic.StationaryProblem`, we can use
a predifined *discretizer* to do the work for us. In this case, we use
:func:`~pymor.discretizers.cg.discretize_stationary_cg`:

>>> fom, fom_data = discretize_stationary_cg(p, diameter=1./100.)

``fom`` is the |StationaryModel| which has been created for us,
whereas ``fom_data`` contains some additional data, in particular the |Grid|
and the |BoundaryInfo| which have been created during discretization. We
can have a look at the grid,

>>> print(fom_data['grid'])
Tria-Grid on domain [0,1] x [0,1]
x0-intervals: 100, x1-intervals: 100
elements: 40000, edges: 60200, vertices: 20201

and, as always, we can display its class documentation using
``help(fom_data['grid'])``.

Let's solve the thermal block problem and visualize the solution:

>>> U = fom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
>>> fom.visualize(U, title='Solution')
00:35 StationaryModel: Solving ThermalBlock((3, 2))_CG for {diffusion: [1.0, 0.1, 0.3, 0.1, 0.2, 1.0]} ...

Each class in pyMOR that describes a |Parameter| dependent mathematical
object, like the |StationaryModel| in our case, derives from
|Parametric| and determines the |Parameters| it expects during :meth:`__init__`
by calling :meth:`~pymor.parameters.base.Parametric.build_parameter_type`.
The resulting |ParameterType| is stored in the object's
:attr:`~pymor.parameters.base.Parametric.parameter_type` attribute. Let us
have a look:

>>> print(fom.parameter_type)
{diffusion: (2, 3)}

This tells us, that the |Parameter| which
:meth:`~pymor.models.interfaces.ModelInterface.solve` expects
should be a dictionary with one key ``'diffusion'`` whose value is a
|NumPy array| of shape ``(2, 3)``, corresponding to the block structure of
the problem. However, by using the
:meth:`~pymor.parameters.base.Parametric.parse_parameter` method, pyMOR is
smart enough to correctly parse the input ``[1.0, 0.1, 0.3, 0.1, 0.2, 1.0]``.

Next we want to use the :func:`~pymor.algorithms.greedy.greedy` algorithm
to reduce the problem. For this we need to choose a reductor which will keep
track of the reduced basis and perform the actual RB-projection. We will use
:class:`~pymor.reductors.coercive.CoerciveRBReductor`, which will
also assemble an error estimator to estimate the reduction error. This
will significantly speed up the basis generation, as we will only need to
solve the high-dimensional problem for those parameters in the training set
which are actually selected for basis extension. To control the condition of
the reduced system matrix, we must ensure that the generated basis is
orthonormal w.r.t. the H1_0-product on the solution space. For this we pass
the :attr:`h1_0_semi_product` attribute of the model as inner product to
the reductor, which will also use it for computing the Riesz representatives
required for error estimation. Moreover, we have to provide
the reductor with a |ParameterFunctional| which computes a lower bound for
the coercivity of the problem for a given parameter.

>>> reductor = CoerciveRBReductor(
...     fom,
...     product=fom.h1_0_semi_product,
...     coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameter_type)
... )

Moreover, we need to select a |Parameter| training set. The model
``fom`` already comes with a |ParameterSpace| which it has inherited from the
analytical problem. We can sample our parameters from this space, which is a
:class:`~pymor.parameters.spaces.CubicParameterSpace`. E.g.:

>>> training_set = fom.parameter_space.sample_uniformly(4)
>>> print(training_set[0])
{diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

Now we start the basis generation:

>>> greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=32)
02:14 weak_greedy: Started greedy search on training set of size 4096.
02:14 weak_greedy: Estimating errors ...
02:14 |   RBSurrogate: Reducing ...
02:14 |   |   CoerciveRBReductor: Operator projection ...
02:14 |   |   CoerciveRBReductor: Assembling error estimator ...
02:14 |   |   |   ResidualReductor: Estimating residual range ...
02:14 |   |   |   |   estimate_image_hierarchical: Estimating image for basis vector -1 ...
02:14 |   |   |   |   estimate_image_hierarchical: Orthonormalizing ...
02:14 |   |   |   ResidualReductor: Projecting residual operator ...
02:14 |   |   CoerciveRBReductor: Building ROM ...
02:16 weak_greedy: Maximum error after 0 extensions: 1.874573182151557 (mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]})
02:16 weak_greedy: Extending surrogate for mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]} ...
02:16 |   RBSurrogate: Computing solution snapshot for mu = {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]} ...
02:16 |   |   StationaryModel: Solving ThermalBlock((3, 2))_CG for {diffusion: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]} ...
02:16 |   RBSurrogate: Extending basis with solution snapshot ...
02:16 |   RBSurrogate: Reducing ...
02:16 |   |   CoerciveRBReductor: Operator projection ...
02:16 |   |   CoerciveRBReductor: Assembling error estimator ...
02:16 |   |   |   ResidualReductor: Estimating residual range ...
02:16 |   |   |   |   estimate_image_hierarchical: Estimating image for basis vector 0 ...
02:16 |   |   |   |   estimate_image_hierarchical: Orthonormalizing ...
02:16 |   |   |   |   |   gram_schmidt: Removing vector 1 of norm 0.0
02:16 |   |   |   |   |   gram_schmidt: Removing linearly dependent vector 7
02:17 |   |   |   ResidualReductor: Projecting residual operator ...
02:17 |   |   CoerciveRBReductor: Building ROM ...
                 ...
                 ...
03:59 weak_greedy: Estimating errors ...
04:02 weak_greedy: Maximum error after 31 extensions: 0.031036281563735547 (mu = {diffusion: [0.1, 0.4, 0.1, 1.0, 0.4, 0.1]})
04:02 weak_greedy: Extending surrogate for mu = {diffusion: [0.1, 0.4, 0.1, 1.0, 0.4, 0.1]} ...
04:02 |   RBSurrogate: Computing solution snapshot for mu = {diffusion: [0.1, 0.4, 0.1, 1.0, 0.4, 0.1]} ...
04:02 |   |   StationaryModel: Solving ThermalBlock((3, 2))_CG for {diffusion: [0.1, 0.4, 0.1, 1.0, 0.4, 0.1]} ...
04:02 |   RBSurrogate: Extending basis with solution snapshot ...
04:02 |   |   gram_schmidt: Orthonormalizing vector 31 again
04:02 |   RBSurrogate: Reducing ...
04:02 |   |   CoerciveRBReductor: Operator projection ...
04:02 |   |   CoerciveRBReductor: Assembling error estimator ...
04:02 |   |   |   ResidualReductor: Estimating residual range ...
04:02 |   |   |   |   estimate_image_hierarchical: Estimating image for basis vector 31 ...
04:02 |   |   |   |   estimate_image_hierarchical: Orthonormalizing ...
04:02 |   |   |   |   |   gram_schmidt: Removing vector 180 of norm 1.670179097624966e-15
04:02 |   |   |   |   |   gram_schmidt: Orthonormalizing vector 181 again
04:02 |   |   |   |   |   gram_schmidt: Orthonormalizing vector 182 again
04:02 |   |   |   |   |   gram_schmidt: Orthonormalizing vector 183 again
04:03 |   |   |   |   |   gram_schmidt: Orthonormalizing vector 184 again
04:03 |   |   |   |   |   gram_schmidt: Orthonormalizing vector 185 again
04:03 |   |   |   |   |   gram_schmidt: Orthonormalizing vector 186 again
04:03 |   |   |   ResidualReductor: Projecting residual operator ...
04:03 |   |   CoerciveRBReductor: Building ROM ...
04:03 weak_greedy: Maximum number of 32 extensions reached.
04:03 weak_greedy: Greedy search took 109.13336181640625 seconds



The ``max_extensions`` parameter defines how many basis vectors we want to
obtain. ``greedy_data`` is a dictionary containing various data that has
been generated during the run of the algorithm:

>>> print(greedy_data.keys())
dict_keys(['rom', 'max_errs', 'extensions', 'max_err_mus', 'time'])

The most important items is ``'rom'`` which holds the reduced |Model|
obtained from applying our reductor with the final reduced basis.

>>> rom = greedy_data['rom']

All vectors in pyMOR are stored in so called |VectorArrays|. For example
the solution ``U`` computed above is given as a |VectorArray| of length 1.
For the reduced basis we have:

>>> RB = reductor.bases['RB']
>>> print(type(RB))
<class 'pymor.vectorarrays.numpy.NumpyVectorArray'>
>>> print(len(RB))
32
>>> print(RB.dim)
20201

Let us check if the reduced basis really is orthonormal with respect to
the H1-product. For this we use the :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
method:

>>> import numpy as np
>>> gram_matrix = RB.gramian(fom.h1_0_semi_product)
>>> print(np.max(np.abs(gram_matrix - np.eye(32))))
1.3736926228686067e-13

Looks good! We can now solve the reduced model for the same parameter as above.
The result is a vector of coefficients w.r.t. the reduced basis, which is
currently stored in ``rb``. To form the linear combination, we can use the
`reconstruct` method of the reductor:

>>> u = rom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
>>> print(u)
[[ 5.66075162e-01 -1.33690839e-01  2.42900421e-01 -1.26735678e-01
  -1.50500532e-01 -5.44158631e-02  1.05816850e-01 -5.18375437e-02
   1.25819087e-01  4.47997587e-02  1.97036013e-01  2.16171955e-02
   1.47925585e-02 -1.20307244e-02 -3.45979550e-02 -1.17557331e-02
  -1.87134380e-02 -6.38947282e-03 -1.41062909e-02  1.82350035e-02
  -8.22668992e-04  8.07669959e-03 -1.65145068e-03 -4.26535407e-03
  -3.03207875e-04 -4.79783929e-03 -1.97741447e-03  7.72000615e-03
   4.05504068e-03  1.21861875e-03  3.82238121e-03 -5.76733834e-03]]
>>> U_red = reductor.reconstruct(u)
>>> print(U_red.dim)
20201

Finally we compute the reduction error and display the reduced solution along with
the detailed solution and the error:

>>> ERR = U - U_red
>>> print(ERR.norm(fom.h1_0_semi_product))
[0.00536024]
>>> fom.visualize((U, U_red, ERR),
...               legend=('Detailed', 'Reduced', 'Error'),
...               separate_colorbars=True)

We can nicely observe that, as expected, the error is maximized along the
jumps of the diffusion coefficient.


Learning more
-------------

As a next step, you should read our :ref:`technical_overview` which discusses the
most important concepts and design decisions behind pyMOR. After that
you should be ready to delve into the reference documentation.

Should you have any problems regarding pyMOR, questions or
`feature requests <https://github.com/pymor/pymor/issues>`_, do not hesitate
to contact us at our
`mailing list <http://listserv.uni-muenster.de/mailman/listinfo/pymor-dev>`_!
