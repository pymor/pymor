.. _getting_started:

***************
Getting started
***************

.. code-links::
    :timeout: -1

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

.. nbplot::
   from pymor.basic import *
   from pymor.core.logger import set_log_levels
   set_log_levels({'pymor.algorithms.greedy': 'ERROR', 'pymor.algorithms.gram_schmidt.gram_schmidt': 'ERROR', 'pymor.algorithms.image.estimate_image_hierarchical': 'ERROR'})

Next we will instantiate a class describing the analytical problem
we want so solve. In this case, a
:meth:`~pymor.analyticalproblems.thermalblock.thermal_block_problem`:

.. nbplot::
   p = thermal_block_problem(num_blocks=(3, 2))

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

.. nbplot::
   fom, fom_data = discretize_stationary_cg(p, diameter=1./50.)

``fom`` is the |StationaryModel| which has been created for us,
whereas ``fom_data`` contains some additional data, in particular the |Grid|
and the |BoundaryInfo| which have been created during discretization. We
can have a look at the grid,

.. nbplot::
   print(fom_data['grid'])

and, as always, we can display its class documentation using
``help(fom_data['grid'])``.

Let's solve the thermal block problem and visualize the solution:

.. nbplot::
   U = fom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
   fom.visualize(U, title='Solution')

Each class in pyMOR that describes a |Parameter| dependent mathematical
object, like the |StationaryModel| in our case, derives from
|Parametric| and determines the |Parameters| it expects during :meth:`__init__`
by calling :meth:`~pymor.parameters.base.Parametric.build_parameter_type`.
The resulting |ParameterType| is stored in the object's
:attr:`~pymor.parameters.base.Parametric.parameter_type` attribute. Let us
have a look:

.. nbplot::
   print(fom.parameter_type)

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

.. nbplot::
   reductor = CoerciveRBReductor(
       fom,
       product=fom.h1_0_semi_product,
       coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameter_type)
   )

Moreover, we need to select a |Parameter| training set. The model
``fom`` already comes with a |ParameterSpace| which it has inherited from the
analytical problem. We can sample our parameters from this space, which is a
:class:`~pymor.parameters.spaces.CubicParameterSpace`. E.g.:

.. nbplot::
   training_set = fom.parameter_space.sample_uniformly(4)
   print(training_set[0])

Now we start the basis generation:

.. nbplot::
  >>> greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=32)

The ``max_extensions`` parameter defines how many basis vectors we want to
obtain. ``greedy_data`` is a dictionary containing various data that has
been generated during the run of the algorithm:

.. nbplot::
   print(greedy_data.keys())

The most important items is ``'rom'`` which holds the reduced |Model|
obtained from applying our reductor with the final reduced basis.

.. nbplot::
   rom = greedy_data['rom']

All vectors in pyMOR are stored in so called |VectorArrays|. For example
the solution ``U`` computed above is given as a |VectorArray| of length 1.
For the reduced basis we have:

.. nbplot::
   RB = reductor.bases['RB']
   print(type(RB))
   print(len(RB))
   print(RB.dim)

Let us check if the reduced basis really is orthonormal with respect to
the H1-product. For this we use the :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
method:

.. nbplot::
   import numpy as np
   gram_matrix = RB.gramian(fom.h1_0_semi_product)
   print(np.max(np.abs(gram_matrix - np.eye(32))))

Looks good! We can now solve the reduced model for the same parameter as above.
The result is a vector of coefficients w.r.t. the reduced basis, which is
currently stored in ``rb``. To form the linear combination, we can use the
`reconstruct` method of the reductor:

.. nbplot::
   u = rom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
   print(u)
   U_red = reductor.reconstruct(u)
   print(U_red.dim)

Finally we compute the reduction error and display the reduced solution along with
the detailed solution and the error:

.. nbplot::
   ERR = U - U_red
   print(ERR.norm(fom.h1_0_semi_product))
   fom.visualize((U, U_red, ERR),
                 legend=('Detailed', 'Reduced', 'Error'),
                 separate_colorbars=True)

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
