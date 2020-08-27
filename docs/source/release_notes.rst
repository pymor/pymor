.. _release_notes:

*************
Release Notes
*************

pyMOR 2020.2 (?, 2020)
----------------------


Release highlights
^^^^^^^^^^^^^^^^^^


Additional new features
^^^^^^^^^^^^^^^^^^^^^^^


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Renaming of some Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~
`ComponentProjection`, `Concatenation` and `LinearAdvectionLaxFriedrichs` were
renamed to `ComponentProjectionOperator`, `ConcatenationOperator` and
`LinearAdvectionLaxFriedrichsOperator`, respectively.


Further notable improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


pyMOR 2020.1 (July 23, 2020)
----------------------------
We are proud to announce the release of pyMOR 2020.1! Highlights of this release
are support for non-intrusive model order reduction using artificial neural networks,
the subspace accelerated dominant pole algorithm (SAMDP) and the implicitly restarted
Arnoldi method for eigenvalue computation. Parameter handling in pyMOR has been
simplified, and a new series of hands-on tutorials helps getting started using pyMOR
more easily.

Over 600 single commits have entered this release. For a full list of changes
see `here <https://github.com/pymor/pymor/compare/2019.2.x...2020.1.x>`__.

pyMOR 2020.1 contains contributions by Linus Balicki, Tim Keil, Hendrik Kleikamp
and Luca Mechelli. We are also happy to welcome Linus as a new main developer!
See `here <https://github.com/pymor/pymor/blob/master/AUTHORS.md>`__ for more
details.


Release highlights
^^^^^^^^^^^^^^^^^^

Model order reduction using artificial neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With this release, we introduce a simple approach for non-intrusive model order
reduction to pyMOR that makes use of artificial neural networks
`[#1001] <https://github.com/pymor/pymor/pull/1001>`_. The method was first
described in [HU18]_ and only requires being able to compute solution snapshots of
the full-order |Model|. Thus, it can be applied to arbitrary (nonlinear) |Models| even when no
access to the model's |Operators| is possible.

Our implementation internally wraps `PyTorch <https://pytorch.org>`_ for the training and evaluation of
the neural networks. No knowledge of PyTorch or neural networks is required to apply the method.


New system analysis and linear algebra algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The new :meth:`~pymor.algorithms.eigs.eigs` method
`[#880] <https://github.com/pymor/pymor/pull/880>`_ computes 
smallest/largest eigenvalues of an arbitary linear real |Operator|
using the implicitly restarted Arnoldi method [RL95]_. It can also
be used to solve generalized eigenvalue problems.

So far, computing poles of an |LTIModel| was only supported by its
:meth:`~pymor.models.iosys.LTIModel.poles` method, which uses a dense eigenvalue
solver and converts the operators to dense matrices.
The new :meth:`~pymor.algorithms.samdp.samdp` method
`[#834] <https://github.com/pymor/pymor/pull/834>`_ implements the
subspace accelerated dominant pole (SAMDP) algorithm  [RM06]_,
which can be used to compute the dominant poles operators of an 
|LTIModel| with arbitrary (in particular sparse) system |Operators|
without relying on dense matrix operations.


Improved parameter handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~
While pyMOR always had a powerful and flexible system for handling |parameters|,
understanding this system was often a challenge for pyMOR newcomers. Therefore,
we have completely overhauled parameter handling in pyMOR, removing some unneeded
complexities and making the nomenclature more straightforward. In particular:

- The `Parameter` class has been renamed to :class:`~pymor.parameters.base.Mu`.
  `ParameterType` has been renamed to |Parameters|. The items of a |Parameters|
  dict are the individual *parameters* of the corresponding |ParametricObject|.
  The items of a :class:`~pymor.parameters.base.Mu` dict are the associated
  *parameter values*.
- All parameters are now one-dimensional NumPy arrays.
- Instead of manually calling `build_parameter_type` in `__init__`, the |parameters|
  of a |ParametricObject| are now automatically inferred from the object's `__init__`
  arguments. The process can be customized using the new `parameters_own` and
  `parameters_internal` properties.
- `CubicParameterSpace` was renamed to |ParameterSpace| and is created using
  `parametric_object.parameters.space(ranges)`.

Further details can be found in `[#923] <https://github.com/pymor/pymor/pull/923>`_.
Also see `[#949] <https://github.com/pymor/pymor/pull/949>`_ and
`[#998] <https://github.com/pymor/pymor/pull/998>`_.


pyMOR tutorial collection
~~~~~~~~~~~~~~~~~~~~~~~~~
Hands-on tutorials provide a good opportunity for new users to get started with
a software library. In this release a variety of tutorials have been added which
introduce important pyMOR concepts and basic model order reduction methods. In
particular users can now learn about:

- :doc:`tutorial_builtin_discretizer`.
- :doc:`tutorial_basis_generation`
- :doc:`tutorial_bt`
- :doc:`tutorial_mor_with_anns`
- :doc:`tutorial_external_solver`


Additional new features
^^^^^^^^^^^^^^^^^^^^^^^

Improvements to ParameterFunctionals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Several improvements have been made to pyMOR's |ParameterFunctionals|:

- `[#934] [parameters/functionals] Add derivative of products <https://github.com/pymor/pymor/pull/934>`_
- `[#950] [parameters/functionals] Add LincombParameterFunctional <https://github.com/pymor/pymor/pull/950>`_
- `[#959] verbose name for d_mu functionals <https://github.com/pymor/pymor/pull/959>`_
- `[#861] Min-theta approach <https://github.com/pymor/pymor/pull/861>`_
- `[#952] add BaseMaxThetaParameterFunctional to generalize max-theta approach  <https://github.com/pymor/pymor/pull/952>`_


Extended Newton algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~
Finding a proper parameter for the step size in the Newton algorithm can be a difficult
task. In this release an Armijo line search algorithm is added which allows for computing
adequate step sizes in every step of the iteration. Details about the line search
implementation in pyMOR can be found in `[#925] <https://github.com/pymor/pymor/pull/925>`_.

Additionally, new options for determining convergence of the Newton method have been added.
It is now possible to choose between the norm of the residual or the update vector as a
measure for the error. Information about other noteworthy improvements that are related to
this change can be found in `[#956] <https://github.com/pymor/pymor/pull/956>`_, as well as
`[#932] <https://github.com/pymor/pymor/pull/932>`_.


initial_guess parameter for apply_inverse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :meth:`~pymor.operators.interface.Operator.apply_inverse` and
:meth:`~pymor.operators.interface.Operator.apply_inverse_adjoint` methods of the |Operator| interface
have gained an additional `initial_guess` parameter that can be passed to iterative linear solvers.
For nonlinear |Operators| the initial guess is passed to the :meth:`~pymor.algorithms.newton.newton`
algorithm `[#941] <https://github.com/pymor/pymor/pull/941>`_.


manylinux 2010+2014 wheels
~~~~~~~~~~~~~~~~~~~~~~~~~~
In addition to `manylinux1 <https://www.python.org/dev/peps/pep-0513/>`_ wheels we are now also shipping wheels
conforming with the `manylinux2010 <https://www.python.org/dev/peps/pep-0571/>`_ and
`manylinux2014 <https://www.python.org/dev/peps/pep-0599/>`_ standards. The infrastructure for this was added in
`[#846] <https://github.com/pymor/pymor/pull/846>`_.


Debugging improvements
~~~~~~~~~~~~~~~~~~~~~~
The :meth:`~pymor.core.defaults.defaults` decorator has been refactored to make stepping through it
with a debugger faster `[#864] <https://github.com/pymor/pymor/pull/864>`_. Similar improvements
have been made to :meth:`RuleTable.apply <pymor.algorithms.rules.RuleTable.apply>`. The new
:meth:`~pymor.algorithms.rules.RuleTable.breakpoint_for_obj` and
:meth:`~pymor.algorithms.rules.RuleTable.breakpoint_for_name` methods allow setting conditional
breakpoints in :meth:`RuleTable.apply <pymor.algorithms.rules.RuleTable.apply>` that match
specific objects to which the table might be applied `[#945] <https://github.com/pymor/pymor/pull/945>`_.


WebGL-based visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~
This release enables our `pythreejs <https://github.com/jupyter-widgets/pythreejs>`_-based visualization module
for Jupyter Notebook environments by default. It acts as a drop in replacement for the previous default, which was
matplotlib based. This new module improves interactive performance for visualizations
with a large number of degrees of freedom by utilizing the user's graphics card via the browser's WebGL API.
The old behaviour can be reactivated using

.. jupyter-execute::

    from pymor.basic import *
    set_defaults({'pymor.discretizers.builtin.gui.jupyter.get_visualizer.backend': 'MPL'})


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Renamed interface classes
~~~~~~~~~~~~~~~~~~~~~~~~~
The names of pyMOR's interface classes have been shortened
`[#859] <https://github.com/pymor/pymor/pull/859>`_.  In particular:

- `VectorArrayInterface`, `OperatorInterface`, `ModelInterface` were renamed to
  |VectorArray|, |Operator|, |Model|. The corresponding modules were renamed from
  `pymor.*.interfaces` to `pymor.*.interface`.
- `BasicInterface`, `ImmutableInterface`, `CacheableInterface` were renamed to
  |BasicObject|, |ImmutableObject|, |CacheableObject|. `pymor.core.interfaces` has
  been renamed to :mod:`pymor.core.base`.

The base classes `OperatorBase`, `ModelBase`, `FunctionBase` were merged into
their respective interface classes `[#859] <https://github.com/pymor/pymor/pull/859>`_,
`[#867] <https://github.com/pymor/pymor/pull/867>`_.


Module cleanup
~~~~~~~~~~~~~~
Modules associated with pyMOR's builtin discretization toolkit were moved to the
:mod:`pymor.discretizers.builtin` package `[#847] <https://github.com/pymor/pymor/pull/847>`_.
The `domaindescriptions` and `functions` packages were made sub-packages of
:mod:`pymor.analyticalproblems` `[#855] <https://github.com/pymor/pymor/pull/855>`_,
`[#858] <https://github.com/pymor/pymor/pull/858>`_. The obsolete code in
`pymor.discretizers.disk` was removed `[#856] <https://github.com/pymor/pymor/pull/856>`_.
Further, the `playground` package was removed `[#940] <https://github.com/pymor/pymor/pull/940>`_.


State ids removed and caching simplified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The unnecessarily complicated concept of *state ids*, which was used to build cache keys
based on the actual state of a |CacheableObject|, has been completely removed from pyMOR.
Instead, now a `cache_id` has to be manually specified when persistent caching over multiple
program runs is desired `[#841] <https://github.com/pymor/pymor/pull/841>`_.


Further API changes
~~~~~~~~~~~~~~~~~~~
- `[#938] Fix order of parameters in thermalblock_problem <https://github.com/pymor/pymor/pull/938>`_
- `[#980] Set gram_schmidt tolerances in POD to 0 to never truncate pod modes <https://github.com/pymor/pymor/pull/980>`_
- `[#1012] Change POD default rtol and fix analyze_pickle demo for numpy master <https://github.com/pymor/pymor/pull/1012>`_


Further notable improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `[#885] Implement VectorArrayOperator.apply_inverse <https://github.com/pymor/pymor/pull/885>`_
- `[#888] Implement FenicsVectorSpace.from_numpy <https://github.com/pymor/pymor/pull/888>`_
- `[#895] Implement VectorArray.__deepcopy__ via VectorArray.copy(deep=True) <https://github.com/pymor/pymor/pull/895>`_
- `[#905] Add from_files method to SecondOrderModel <https://github.com/pymor/pymor/pull/905>`_
- `[#919] [reductors.coercive] remove unneccessary initialization in SimpleCoerciveReductor <https://github.com/pymor/pymor/pull/919>`_
- `[#926] [Operators] Speed up apply methods for LincombOperator <https://github.com/pymor/pymor/pull/926>`_
- `[#937] Move NumpyListVectorArrayMatrixOperator out of the playground <https://github.com/pymor/pymor/pull/937>`_
- `[#943] [logger] adds a ctx manager that restores effective level on exit <https://github.com/pymor/pymor/pull/943>`_





pyMOR 2019.2 (December 16, 2019)
--------------------------------
We are proud to announce the release of pyMOR 2019.2! For this release we have
worked hard to make implementing new models and reduction algorithms with pyMOR
even easier. Further highlights of this release are an extended VectorArray
interface with generic support for complex numbers, vastly extended and
improved system-theoretic MOR methods, as well as builtin support for model
outputs and parameter sensitivities.

Over 700 single commits have entered this release. For a full list of changes
see `here <https://github.com/pymor/pymor/compare/0.5.x...2019.2.x>`__.

pyMOR 2019.2 contains contributions by Linus Balicki, Dennis Eickhorn and Tim
Keil. See `here <https://github.com/pymor/pymor/blob/master/AUTHORS.md>`__ for
more details.


Release highlights
^^^^^^^^^^^^^^^^^^

Implement new models and reductors more easily
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As many users have been struggling with the notion of `Discretization` in pyMOR
and to account for the fact that not every full-order model needs to be a discretized
PDE model, we have decided to rename `DiscretizationInterface` to
:class:`~pymor.models.interfaces.ModelInterface` and all deriving classes accordingly
`[#568] <https://github.com/pymor/pymor/pull/568>`_. Consequently, the variable names
`m`, `rom`, `fom` will now be found throughout pyMOR's code to refer to an arbitrary
|Model|, a reduced-order |Model| or a full-order |Model|.

Moreover, following the `Zen of Python's <https://www.python.org/dev/peps/pep-0020/>`_
'Explicit is better than implicit' and 'Simple is better than complex', we have
completely revamped the implementation of |Models| and :mod:`~pymor.reductors`
to facilitate the implementation of new model types and reduction methods
`[#592] <https://github.com/pymor/pymor/pull/592>`_. In particular, the complicated
and error-prone approach of trying to automatically correctly project the |Operators|
of any given |Model| in `GenericRBReductor` and `GenericPGReductor` has been replaced
by simple |Model|-adapted reductors which explicitly state with which bases each
|Operator| shall be projected. As a consequence, we could remove the `operators` dict
and the notion of `special_operators` in :class:`~pymor.models.basic.ModelBase`,
vastly simplifying its implementation and the definition of new |Model| classes.


Extended VectorArray interface with generic complex number support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`~pymor.vectorarrays.interfaces.VectorArrayInterface` has been extended to
allow the creation of non-zero vectors using the
:meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.ones` and
:meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.full` methods
`[#612] <https://github.com/pymor/pymor/pull/612>`_. Vectors with random values can
be created using the :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.random`
method `[#618] <https://github.com/pymor/pymor/pull/618>`_. All |VectorArray|
implementations shipped with pyMOR support these new interface methods.
As an important step to improve the support for system-theoretic MOR methods with
external PDE solvers, we have implemented facilities to provide generic support
for complex-valued |VectorArrays| even for PDE solvers that do not support complex
vectors natively `[#755] <https://github.com/pymor/pymor/pull/755>`_.


Improved and extended support for system-theoretic MOR methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To increase compatibility between input-output models in
:mod:`~pymor.models.iosys` and the |InstationaryModel|, support for models with
parametric operators has been added
`[#626] <https://github.com/pymor/pymor/pull/626>`_, which also enables
implementation of parametric MOR methods for such models.
Furthermore, the `state_space` attribute was removed in favor of
`solution_space` `[#648] <https://github.com/pymor/pymor/pull/648>`_ to make
more explicit the result of the
:meth:`~pymor.models.interfaces.ModelInterface.solve` method.
Further improvements in naming has been renaming attributes `n`, `m`, and `p` to
`order`, `input_dim`, and `output_dim`
`[#578] <https://github.com/pymor/pymor/pull/578>`_ and the `bode` method to
:meth:`~pymor.models.iosys.InputOutputModel.freq_resp`
`[#729] <https://github.com/pymor/pymor/pull/729>`_.
Reductors in :mod:`~pymor.reductors.bt` and :mod:`~pymor.reductors.h2` received
numerous improvements (`[#656] <https://github.com/pymor/pymor/pull/656>`_,
`[#661] <https://github.com/pymor/pymor/pull/661>`_,
`[#807] <https://github.com/pymor/pymor/pull/807>`_) and variants of one-sided
IRKA have been added `[#579] <https://github.com/pymor/pymor/pull/579>`_.
As for Lyapunov equations, a low-rank solver for Riccati equations has been
added `[#736] <https://github.com/pymor/pymor/pull/736>`_.


Model outputs and parameter sensitivities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The notion of a |Model|'s output has been formally added to the
:class:`~pymor.models.interfaces.ModelInterface` `[#750] <https://github.com/pymor/pymor/pull/750>`_:
The output of a |Model| is defined to be a |VectorArray| of the model's
:attr:`~pymor.models.interfaces.ModelInterface.output_space` |VectorSpace| and
can be computed using the new :meth:`~pymor.models.interfaces.ModelInterface.output` method.
Alternatively, :meth:`~pymor.models.interfaces.ModelInterface.solve` method can
now be called with `return_output=True` to return the output alongside the state space
solution.

To compute parameter sensitivities, we have added `d_mu` methods to
:meth:`OperatorInterface <pymor.operators.interfaces.OperatorInterface.d_mu>` and
:meth:`ParameterFunctionalInterface <pymor.parameters.interfaces.ParameterFunctionalInterface.d_mu>`
which return the partial derivative with respect to a given parameter component
`[#748] <https://github.com/pymor/pymor/pull/748>`_.


Additional new features
^^^^^^^^^^^^^^^^^^^^^^^

Extended FEniCS bindings
~~~~~~~~~~~~~~~~~~~~~~~~
FEniCS support has been improved by adding support for nonlinear |Operators| including
an implementation of :meth:`~pymor.operators.interfaces.OperatorInterface.restricted`
to enable fast local evaluation of the operator for efficient
:class:`empirical interpolation <pymor.operators.ei.EmpiricalInterpolatedOperator>`
`[#819] <https://github.com/pymor/pymor/pull/819>`_. Moreover the parallel implementations
of :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.amax` and
:meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dofs` have been fixed
`[#616] <https://github.com/pymor/pymor/pull/616>`_ and
:attr:`~pymor.operators.interfaces.OperatorInterface.solver_options` are now correctly
handled in :meth:`~pymor.operators.interfaces.OperatorInterface._assemble_lincomb`
`[#812] <https://github.com/pymor/pymor/pull/812>`_.



Improved greedy algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~
pyMOR's greedy algorithms have been refactored into :func:`~pymor.algorithms.greedy.weak_greedy`
and :func:`~pymor.algorithms.adaptivegreedy.adaptive_weak_greedy` functions that
use a common :class:`~pymor.algorithms.greedy.WeakGreedySurrogate` to estimate
the approximation error and extend the greedy bases. This allows these functions to be
used more flexible, e.g. for goal-oriented basis generation, by implementing a new
:class:`~pymor.algorithms.greedy.WeakGreedySurrogate` `[#757] <https://github.com/pymor/pymor/pull/757>`_.


Numerical linear algebra algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By specifying `return_R=True`, the :func:`~pymor.algorithms.gram_schmidt.gram_schmidt`
algorithm can now also be used to compute a QR decomposition of a given |VectorArray|
`[#577] <https://github.com/pymor/pymor/pull/577>`_. Moreover,
:func:`~pymor.algorithms.gram_schmidt.gram_schmidt` can be used as a more accurate
(but often more expensive) alternative for computing the :func:`~pymor.algorithms.pod.pod` of
a |Vectorarray|. Both, the older method-of-snapshots approach as well as the QR decomposition
are now available for computing a truncated SVD of a |VectorArray| via the newly added
:mod:`~pymor.algorithms.svd_va` module `[#718] <https://github.com/pymor/pymor/pull/718>`_.
Basic randomized algorithms for approximating the image of a linear |Operator| are
implemented in the :mod:`~pymor.algorithms.randrangefinder` module
`[#665] <https://github.com/pymor/pymor/pull/665>`_.


Support for low-rank operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Low-rank |Operators| and as well as sums of arbitrary |Operators| with a low-rank
|Operator| can now be represented by :class:`~pymor.operators.constructions.LowRankOperator`
and :class:`~pymor.operators.constructions.LowRankUpdatedOperator`. For the latter,
:meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse` and
:meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse_adjoint` are implemented
via the Sherman-Morrison-Woodbury formula `[#743] <https://github.com/pymor/pymor/pull/743>`_.


Improved string representations of pyMOR objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Custom  `__str__` special methods have been implemented for all |Model| classes shipped with
pyMOR `[#652] <https://github.com/pymor/pymor/pull/652>`_. Moreover, we have added a generic
`__repr__` implementation to `BasicInterface` which recursively prints all class attributes
corresponding to an `__init__` argument (with a non-default value)
`[#706] <https://github.com/pymor/pymor/pull/706>`_.


Easier working with immutable objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A new check in :class:`~pymor.core.interfaces.ImmutableMeta` enforces all `__init__` arguments
of an |immutable| object to be available as object attributes, thus ensuring that
`~pymor.core.interfaces.ImmutableInterface.with_` works reliably with all |immutable| objects
in pyMOR `[#694] <https://github.com/pymor/pymor/pull/694>`_. To facilitate the initialization
of these attributes in `__init__` the
`__auto_init <https://github.com/pymor/pymor/pull/732/files#diff-9ff4f0e773ee7352ff323cb88a3adeabR149-R164>`_
method has been added to `BasicInterface` `[#732] <https://github.com/pymor/pymor/pull/732>`_.
Finally, `~pymor.core.interfaces.ImmutableInterface.with_` now has a `new_type` parameter
which allows to change the class of the object returned by it
`[#705] <https://github.com/pymor/pymor/pull/705>`_.


project and assemble_lincomb are easier to extend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In pyMOR 0.5, we have introduced |RuleTables| to make central algorithms in
pyMOR, like the projection of an |Operator| via |project|, easier to trace and
extend.
For pyMOR 2019.2, we have further simplified |project| by removing the `product`
argument from the underlying |RuleTable| `[#785] <https://github.com/pymor/pymor/pull/785>`_.
As the inheritance-based implementation of `assemble_lincomb` was showing similar
complexity issues as the old inheritance-based implementation of `projected`, we
moved all backend-agnostic logic into the |RuleTable|-based free function
:func:`~pymor.algorithms.lincomb.assemble_lincomb`, leaving the remaining backend
code in :meth:`~pymor.operators.interfaces.OperatorInterface._assemble_lincomb`
`[#619] <https://github.com/pymor/pymor/pull/619>`_.


Improvements to pyMOR's discretization toolbox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyMOR's builtin discretization toolbox as seen multiple minor improvements:

- `[#821] Enable to have parametric dirichlet in fv <https://github.com/pymor/pymor/pull/821>`_
- `[#687] Discretizing robin boundary conditions on a RectGrid <https://github.com/pymor/pymor/pull/687>`_
- `[#691] Remove 'order' arguments from CG operators <https://github.com/pymor/pymor/pull/691>`_
- `[#760] [discretizers.cg] affine decomposition of robin operator and rhs functionals <https://github.com/pymor/pymor/pull/760>`_
- `[#793] Use meshio for Gmsh file parsing <https://github.com/pymor/pymor/pull/793>`_


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dropped Python 3.5 support
~~~~~~~~~~~~~~~~~~~~~~~~~~
As Python 3.6 or newer now ships with the current versions of all major Linux distributions,
we have decided to drop support for Python 3.6 in pyMOR 2019.2. This allows us to benefit
from new language features, in particular f-strings and class attribute definition order
preservation `[#553] <https://github.com/pymor/pymor/pull/553>`_,
`[#584] <https://github.com/pymor/pymor/pull/553>`_.


Global RandomState
~~~~~~~~~~~~~~~~~~
pyMOR now has a (mutable) global default :class:`~numpy.random.RandomState`. This means
that when :meth:`~pymor.parameters.spaces.CubicParameterSpace.sample_randomly` is called
repeatedly without specifying a `random_state` or `seed` argument, different |Parameter|
samples will be returned in contrast to the (surprising) previous behavior where the
same samples would have been returned. The same :class:`~numpy.random.RandomState` is
used by the newly introduced :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.random`
method of the :class:`~pymor.vectorarrays.interfaces.VectorArrayInterface`
`[#620] <https://github.com/pymor/pymor/pull/620>`_.


Space id handling
~~~~~~~~~~~~~~~~~
The usage of |VectorSpace| :attr:`ids <pymor.vectorarrays.interfaces.VectorSpace.id>` in pyMOR
has been reduced throughout pyMOR to avoid unwanted errors due to incompatible |VectorSpaces|
(that only differ by their id):

- `[#611] [models.iosys] remove space id handling except for factory methods <https://github.com/pymor/pymor/pull/611>`_
- `[#613] Remove VectorSpace id handling from projection methods <https://github.com/pymor/pymor/pull/613>`_
- `[#614] Remove id from BlockVectorSpace <https://github.com/pymor/pymor/pull/614>`_
- `[#615] Remove 'space' parameter from as_vector <https://github.com/pymor/pymor/pull/615>`_


Further API Changes
~~~~~~~~~~~~~~~~~~~
- The stagnation criterion of the :func:`~pymor.algorithms.newton.newton` is disabled by default
  (and a relaxation parameter has been added) `[#800] <https://github.com/pymor/pymor/pull/800>`_.
- The `coordinates` parameter of :class:`~pymor.parameters.functionals.ProjectionParameterFunctional`
  has been renamed to `index` `[#756] <https://github.com/pymor/pymor/pull/756>`_.


Further notable improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `[#559] fix arnoldi when E is not identity <https://github.com/pymor/pymor/pull/559>`_
- `[#569] Fix NonProjectedResidualOperator.apply <https://github.com/pymor/pymor/pull/569>`_
- `[#585] implement MPIOperator.apply_inverse_adjoint <https://github.com/pymor/pymor/pull/585>`_
- `[#607] Replace sqlite caching <https://github.com/pymor/pymor/pull/607>`_
- `[#608] [mpi] small tweaks to make MPI wrapping more flexible <https://github.com/pymor/pymor/pull/608>`_
- `[#627] Fix as_source_array/as_range_array for BlockRowOperator/BlockColumnOperator <https://github.com/pymor/pymor/pull/627>`_
- `[#644] Replace numpy.linalg.solve by scipy.linalg.solve <https://github.com/pymor/pymor/pull/644>`_
- `[#663] [NumpyVectorSpace] fix issue 662 <https://github.com/pymor/pymor/pull/663>`_
- `[#668] Fixed complex norms <https://github.com/pymor/pymor/pull/668>`_
- `[#693] [parameters.functionals] implement __neg__ <https://github.com/pymor/pymor/pull/693>`_
- `[#702] Add 'linear' attribute to StationaryModel and InstationaryModel <https://github.com/pymor/pymor/pull/702>`_
- `[#716] Fix 643 <https://github.com/pymor/pymor/pull/716>`_
- `[#786] Handle projection of parametric BlockOperators <https://github.com/pymor/pymor/pull/786>`_
- `[#789] allow time-dep operator or rhs in ParabolicRBReductor <https://github.com/pymor/pymor/pull/789>`_
- `[#790] Default to POD-Greedy for instationary problems <https://github.com/pymor/pymor/pull/790>`_
- `[#791] Add rule to ProjectRules for the case that source_basis range basis are None <https://github.com/pymor/pymor/pull/791>`_
- `[#802] Fix project call in ProjectedOperator.jacobian() <https://github.com/pymor/pymor/pull/802>`_
- `[#804] Minor improvements to deim algorithm <https://github.com/pymor/pymor/pull/804>`_
- `[#808] Add convergence check for pymess <https://github.com/pymor/pymor/pull/808>`_
- `[#809] Avoid checking in BlockOperators if block is None <https://github.com/pymor/pymor/pull/809>`_
- `[#814] [algorithms.image] fix CollectVectorRangeRules for ConcatenationOperator <https://github.com/pymor/pymor/pull/814>`_
- `[#815] Make assumptions on mass Operator in InstationaryModel consistent <https://github.com/pymor/pymor/pull/815>`_
- `[#824] Fix NumpyVectorArray.__mul__ when other is a NumPy array <https://github.com/pymor/pymor/pull/824>`_
- `[#827] Add Gitlab Pages hosting for docs + introduce nbplots for sphinx <https://github.com/pymor/pymor/pull/827>`_





pyMOR 0.5 (January 17, 2019)
----------------------------

After more than two years of development, we are proud to announce the release
of pyMOR 0.5! Highlights of this release are support for Python 3, bindings for
the NGSolve finite element library, new linear algebra algorithms, various
|VectorArray| usability improvements, as well as a redesign of pyMOR's
projection algorithms based on |RuleTables|.

Especially we would like to highlight the addition of various system-theoretic
reduction methods such as Balanced Truncation or IRKA. All algorithms are
implemented in terms of pyMOR's |Operator| and |VectorArray| interfaces,
allowing their application to any model implemented using one of the PDE solver
supported by pyMOR. In particular, no import of the system matrices is
required.

Over 1,500 single commits have entered this release. For a full list of changes
see `here <https://github.com/pymor/pymor/compare/0.4.x...0.5.x>`__.

pyMOR 0.5 contains contributions by Linus Balicki, Julia Brunken and Christoph
Lehrenfeld. See `here <https://github.com/pymor/pymor/blob/master/AUTHORS.md>`__
for more details.



Release highlights
^^^^^^^^^^^^^^^^^^


Python 3 support
~~~~~~~~~~~~~~~~

pyMOR is now compatible with Python 3.5 or greater. Since the use of Python 3 is
now standard in the scientific computing community and security updates for
Python 2 will stop in less than a year (https://pythonclock.org), we decided to
no longer support Python 2 and make pyMOR 0.5 a Python 3-only release. Switching
to Python 3 also allows us to leverage newer language features such as the `@`
binary operator for concatenation of |Operators|, keyword-only arguments or
improved support for asynchronous programming.



System-theoretic MOR methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With 386 commits, `[#464] <https://github.com/pymor/pymor/pull/464>`_ added
systems-theoretic methods to pyMOR. Module :mod:`pymor.discretizations.iosys`
contains new discretization classes for input-output systems, e.g. `LTISystem`,
`SecondOrderSystem` and |TransferFunction|. At present, methods related to these
classes mainly focus on continuous-time, non-parametric systems.

Since matrix equation solvers are important tools in many system-theoretic
methods, support for Lyapunov, Riccati and Sylvester equations has been added in
:mod:`pymor.algorithms.lyapunov`, :mod:`pymor.algorithms.riccati` and
:mod:`pymor.algorithms.sylvester`. A generic low-rank ADI (Alternating Direction
Implicit) solver for Lyapunov equations is implemented in
:mod:`pymor.algorithms.lradi`. Furthermore, bindings to low-rank and dense
solvers for Lyapunov and Riccati equations from |SciPy|,
`Slycot <https://github.com/python-control/Slycot>`_ and
`Py-M.E.S.S. <https://www.mpi-magdeburg.mpg.de/projects/mess>`_ are provided in
:mod:`pymor.bindings.scipy`, :mod:`pymor.bindings.slycot` and
:mod:`pymor.bindings.pymess`. A generic Schur decomposition-based solver for
sparse-dense Sylvester equations is implemented in
:mod:`pymor.algorithms.sylvester`.

Balancing Truncation (BT) and Iterative Rational Krylov Algorithm (IRKA) are
implemented in :class:`~pymor.reductors.bt.BTReductor` and
:class:`~pymor.reductors.h2.IRKAReductor`. LQG and Bounded Real variants of BT
are also available (:class:`~pymor.reductors.bt.LQGBTReductor`,
:class:`~pymor.reductors.bt.BRBTReductor`). Bitangential Hermite interpolation
(used in IRKA) is implemented in
:class:`~pymor.reductors.interpolation.LTI_BHIReductor`. Two-Sided Iteration
Algorithm (TSIA), a method related to IRKA, is implemented in
:class:`~pymor.reductors.h2.TSIAReductor`.

Several structure-preserving MOR methods for second-order systems have been
implemented. Balancing-based MOR methods are implemented in
:mod:`pymor.reductors.sobt`, bitangential Hermite interpolation in
:class:`~pymor.reductors.interpolation.SO_BHIReductor` and Second-Order Reduced
IRKA (SOR-IRKA) in :class:`~pymor.reductors.sor_irka.SOR_IRKAReductor`.

For more general transfer functions, MOR methods which return `LTISystems` are
also available. Bitangential Hermite interpolation is implemented in
:class:`~pymor.reductors.interpolation.TFInterpReductor` and Transfer Function
IRKA (TF-IRKA) in :class:`~pymor.reductors.h2.TF_IRKAReductor`.

Usage examples can be found in the `heat` and `string_equation` demo scripts.


NGSolve support
~~~~~~~~~~~~~~~

We now ship bindings for the `NGSolve <https://ngsolve.org>`_ finite element
library. Wrapper classes for |VectorArrays| and matrix-based |Operators| can be
found in the :mod:`pymor.bindings.ngsolve` module. A usage example can be found
in the `thermalblock_simple` demo script.


New linear algebra algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pyMOR now includes an implementation of the
`HAPOD algorithm <https://doi.org/10.1137/16M1085413>`_ for fast distributed
or incremental computation of the Proper Orthogonal Decomposition
(:mod:`pymor.algorithms.hapod`). The code allows for arbitrary sub-POD trees,
on-the-fly snapshot generation and shared memory parallelization via
:mod:`concurrent.futures`. A basic usage example can be found in the `hapod`
demo script.

In addition, the Gram-Schmidt biorthogonalization algorithm has been included in
:mod:`pymor.algorithms.gram_schmidt`.


VectorArray improvements
~~~~~~~~~~~~~~~~~~~~~~~~

|VectorArrays| in pyMOR have undergone several usability improvements:

- The somewhat dubious concept of a `subtype` has been superseded by the concept
  of |VectorSpaces| which act as factories for |VectorArrays|. In particular,
  instead of a `subtype`, |VectorSpaces| can now hold meaningful attributes
  (e.g. the dimension) which are required to construct |VectorArrays| contained
  in the space. The
  :attr:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.id` attribute
  allows to differentiate between technically identical but mathematically
  different spaces `[#323] <https://github.com/pymor/pymor/pull/323>`_.

- |VectorArrays| can now be indexed to select a subset of vectors to operate on.
  In contrast to advanced indexing in |NumPy|, indexing a |VectorArray| will
  always return a view onto the original array data
  `[#299] <https://github.com/pymor/pymor/pull/299>`_.

- New methods with clear semantics have been introduced for the conversion of
  |VectorArrays| to
  (:meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.to_numpy`) and
  from (:meth:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.from_numpy`)
  |NumPy arrays| `[#446] <https://github.com/pymor/pymor/pull/446>`_.

- Inner products between |VectorArrays| w.r.t. to a given inner product
  |Operator| or their norm w.r.t. such an operator can now easily be computed by
  passing the |Operator| as the optional `product` argument to the new
  :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.inner` and
  :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.norm` methods
  `[#407] <https://github.com/pymor/pymor/pull/407>`_.

- The `components` method of |VectorArrays| has been renamed to the more
  intuitive name
  :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dofs`
  `[#414] <https://github.com/pymor/pymor/pull/414>`_.

- The :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.l2_norm2` and
  :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.norm2` have been
  introduced to compute the squared vector norms
  `[#237] <https://github.com/pymor/pymor/pull/237>`_.



RuleTable based algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~

In pyMOR 0.5, projection algorithms are implemented via recursively applied
tables of transformation rules. This replaces the previous inheritance-based
approach. In particular, the `projected` method to perform a (Petrov-)Galerkin
projection of an arbitrary |Operator| has been removed and replaced by a free
|project| function. Rule-based algorithms are implemented by deriving from the
|RuleTable| base class `[#367] <https://github.com/pymor/pymor/pull/367>`_,
`[#408] <https://github.com/pymor/pymor/pull/408>`_.

This approach has several advantages:

- Rules can match based on the class of the object, but also on more general
  conditions, e.g. the name of the |Operator| or being linear and
  non-|parametric|.
- The entire mathematical algorithm can be specified in a single file even when
  the definition of the possible classes the algorithm can be applied to is
  scattered over various files.
- The precedence of rules is directly apparent from the definition of the
  |RuleTable|.
- Generic rules (e.g. the projection of a linear non-|parametric| |Operator| by
  simply applying the basis) can be easily scheduled to take precedence over
  more specific rules.
- Users can implement or modify |RuleTables| without modification of the classes
  shipped with pyMOR.



Additional new features
^^^^^^^^^^^^^^^^^^^^^^^

- Reduction algorithms are now implemented using mutable reductor objects, e.g.
  :class:`~pymor.reductors.basic.GenericRBReductor`, which store and
  :meth:`extend <pymor.reductors.basic.GenericRBReductor.extend_basis>` the
  reduced bases onto which the model is projected. The only return value of the
  reductor's :meth:`~pymor.reductors.basic.GenericRBReductor.reduce` method is
  now the reduced discretization. Instead of a separate reconstructor, the
  reductor's :meth:`~pymor.reductors.basic.GenericRBReductor.reconstruct` method
  can be used to reconstruct a high-dimensional state-space representation.
  Additional reduction data (e.g. used to speed up repeated reductions in greedy
  algorithms) is now managed by the reductor
  `[#375] <https://github.com/pymor/pymor/pull/375>`_.

- Linear combinations and concatenations of |Operators| can now easily be formed
  using arithmetic operators `[#421] <https://github.com/pymor/pymor/pull/421>`_.

- The handling of complex numbers in pyMOR is now more consistent. See
  `[#458] <https://github.com/pymor/pymor/pull/459>`_,
  `[#362] <https://github.com/pymor/pymor/pull/362>`_,
  `[#447] <https://github.com/pymor/pymor/pull/447>`_
  for details. As a consequence of these changes, the `rhs` |Operator| in
  `StationaryDiscretization` is now a vector-like |Operator| instead of a functional.

- The analytical problems and discretizers of pyMOR's discretization toolbox
  have been reorganized and improved. All problems are now implemented as
  instances of |StationaryProblem| or |InstationaryProblem|, which allows an
  easy exchange of data |Functions| of a predefined problem with user-defined
  |Functions|. Affine decomposition of |Functions| is now represented by
  specifying a :class:`~pymor.functions.basic.LincombFunction` as the respective
  data function
  `[#312] <https://github.com/pymor/pymor/pull/312>`_,
  `[#316] <https://github.com/pymor/pymor/pull/316>`_,
  `[#318] <https://github.com/pymor/pymor/pull/318>`_,
  `[#337] <https://github.com/pymor/pymor/pull/337>`_.

- The :mod:`pymor.core.config` module allows simple run-time checking of the
  availability of optional dependencies and their versions
  `[#339] <https://github.com/pymor/pymor/pull/339>`_.

- Packaging improvements

  A compiler toolchain is no longer necessary to install pyMOR as we are now
  distributing binary wheels for releases through the Python Package Index
  (PyPI). Using the `extras_require` mechanism the user can select to install
  either a minimal set::

    pip install pymor

  or almost all, including optional, dependencies::

    pip install pymor[full]

  A docker image containing all of the discretization packages pyMOR has
  bindings to is available for demonstration and development purposes::

    docker run -it pymor/demo:0.5 pymor-demo -h
    docker run -it pymor/demo:0.5 pymor-demo thermalblock --fenics 2 2 5 5



Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `dim_outer` has been removed from the grid interface `[#277]
  <https://github.com/pymor/pymor/pull/277>`_.

- All wrapper code for interfacing with external PDE libraries or equation
  solvers has been moved to the :mod:`pymor.bindings` package. For instance,
  `FenicsMatrixOperator` can now be found in the :mod:`pymor.bindings.fenics`
  module. `[#353] <https://github.com/pymor/pymor/pull/353>`_

- The `source` and `range` arguments of the constructor of
  :class:`~pymor.operators.constructions.ZeroOperator` have
  been swapped to comply with related function signatures
  `[#415] <https://github.com/pymor/pymor/pull/415>`_.

- The identifiers `discretization`, `rb_discretization`, `ei_discretization`
  have been replaced by `d`, `rd`, `ei_d` throughout pyMOR
  `[#416] <https://github.com/pymor/pymor/pull/416>`_.

- The `_matrix` attribute of |NumpyMatrixOperator| has been renamed to `matrix`
  `[#436] <https://github.com/pymor/pymor/pull/436>`_. If `matrix` holds a
  |NumPy array| this array is automatically made read-only to prevent accidental
  modification of the |Operator| `[#462] <https://github.com/pymor/pymor/pull/462>`_.

- The `BoundaryType` class has been removed in favor of simple strings `[#305]
  <https://github.com/pymor/pymor/pull/305>`_.

- The complicated and unused mapping of local parameter component names to
  global names has been removed `[#306] <https://github.com/pymor/pymor/pull/306>`_.



Further notable improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `[#176] Support different colormaps in GLPatchWidget <https://github.com/pymor/pymor/pull/176>`_.
- `[#238] From Operator to NumPy operator <https://github.com/pymor/pymor/pull/238>`_.
- `[#308] Add NumpyGenericOperator.apply_adjoint <https://github.com/pymor/pymor/pull/308>`_.
- `[#313] Add finiteness checks to linear solvers <https://github.com/pymor/pymor/pull/313>`_.
- `[#314] [ExpressionFunction] add components of mu to locals <https://github.com/pymor/pymor/pull/314>`_.
- `[#315] [functions] some improvements to ExpressionFunction/GenericFunction <https://github.com/pymor/pymor/pull/315>`_.
- `[#338] Do not print version string on import <https://github.com/pymor/pymor/pull/338>`_.
- `[#346] Implement more arithmetic operations on VectorArrays and Operators <https://github.com/pymor/pymor/pull/346>`_.
- `[#348] add InverseOperator and InverseTransposeOperator <https://github.com/pymor/pymor/pull/348>`_.
- `[#359] [grids] bugfix for boundary handling in subgrid <https://github.com/pymor/pymor/pull/359>`_.
- `[#365] [operators] add ProxyOperator <https://github.com/pymor/pymor/pull/365>`_.
- `[#366] [operators] add LinearOperator and AffineOperator <https://github.com/pymor/pymor/pull/366>`_.
- `[#368] Add support for PyQt4 and PyQt5 by using Qt.py shim <https://github.com/pymor/pymor/pull/368>`_.
- `[#369] Add basic support for visualization in juypter notebooks <https://github.com/pymor/pymor/pull/369>`_.
- `[#370] Let BitmapFunction accept non-grayscale images <https://github.com/pymor/pymor/pull/370>`_.
- `[#382] Support mpi4py > 2.0 <https://github.com/pymor/pymor/pull/382>`_.
- `[#401] [analyticalproblems] add text_problem <https://github.com/pymor/pymor/pull/401>`_.
- `[#410] add relative_error and project_array functions <https://github.com/pymor/pymor/pull/410>`_.
- `[#422] [Concatenation] allow more than two operators in a Concatenation <https://github.com/pymor/pymor/pull/422>`_.
- `[#425] [ParameterType] base implementation on OrderedDict <https://github.com/pymor/pymor/pull/425>`_.
- `[#431] [operators.cg] fix first order integration <https://github.com/pymor/pymor/pull/431>`_.
- `[#437] [LincombOperator] implement 'apply_inverse' <https://github.com/pymor/pymor/pull/437>`_.
- `[#438] Fix VectorArrayOperator, generalize as_range/source_array <https://github.com/pymor/pymor/pull/438>`_.
- `[#441] fix #439 (assemble_lincomb "operators" parameter sometimes list, sometimes tuple) <https://github.com/pymor/pymor/pull/441>`_.
- `[#452] Several improvements to pymor.algorithms.ei.deim <https://github.com/pymor/pymor/pull/452>`_.
- `[#453] Extend test_assemble <https://github.com/pymor/pymor/pull/453>`_.
- `[#480| [operators] Improve subtraction of LincombOperators <https://github.com/pymor/pymor/pull/480>`_.
- `[#481] [project] ensure solver_options are removed from projected operators <https://github.com/pymor/pymor/pull/481>`_.
- `[#484] [docs] move all references to bibliography.rst <https://github.com/pymor/pymor/pull/484>`_.
- `[#488] [operators.block] add BlockRowOperator, BlockColumnOperator <https://github.com/pymor/pymor/pull/488>`_.
- `[#489] Output functionals in CG discretizations <https://github.com/pymor/pymor/pull/489>`_.
- `[#497] Support automatic conversion of InstationaryDiscretization to LTISystem <https://github.com/pymor/pymor/pull/497>`_.




pyMOR 0.4 (September 28, 2016)
------------------------------

With the pyMOR 0.4 release we have changed the copyright of
pyMOR to

  Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.

Moreover, we have added a `Contribution guideline <https://github.com/pymor/pymor/blob/master/CONTRIBUTING.md>`_
to help new users with starting to contribute to pyMOR.
Over 800 single commits have entered this release.
For a full list of changes see
`here <https://github.com/pymor/pymor/compare/0.3.2...0.4.x>`__.
pyMOR 0.4 contains contributions by Andreas Buhr, Michael Laier, Falk Meyer,
Petar Mlinarić and Michael Schaefer. See
`here <https://github.com/pymor/pymor/blob/master/AUTHORS.md>`__ for more
details.


Release highlights
^^^^^^^^^^^^^^^^^^

FEniCS and deal.II support
~~~~~~~~~~~~~~~~~~~~~~~~~~
pyMOR now includes wrapper classes for integrating PDE solvers
written with the `dolfin` library of the `FEniCS <https://fenicsproject.org>`_
project. For a usage example, see :meth:`pymordemos.thermalblock_simple.discretize_fenics`.
Experimental support for `deal.II <http://dealii.org>`_ can be
found in the `pymor-deal.II <https://github.com/pymor/pymor-deal.II>`_
repository of the pyMOR GitHub organization.


Parallelization of pyMOR's reduction algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We have added a parallelization framework to pyMOR which allows
parallel execution of reduction algorithms based on a simple
|WorkerPool| interface `[#14] <https://github.com/pymor/pymor/issues/14>`_.
The :meth:`~pymor.algorithms.greedy.greedy` `[#155] <https://github.com/pymor/pymor/pull/155>`_
and :meth:`~pymor.algorithms.ei.ei_greedy` algorithms `[#162] <https://github.com/pymor/pymor/pull/162>`_
have been refactored to utilize this interface.
Two |WorkerPool| implementations are shipped with pyMOR:
:class:`~pymor.parallel.ipython.IPythonPool` utilizes the parallel
computing features of `IPython <https://ipython.org/>`_, allowing
parallel algorithm execution in large heterogeneous clusters of
computing nodes. :class:`~pymor.parallel.mpi.MPIPool` can be used
to benefit from existing MPI-based parallel HPC computing architectures
`[#161] <https://github.com/pymor/pymor/issues/161>`_.


Support classes for MPI distributed external PDE solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While pyMOR's |VectorArray|, |Operator| and `Discretization`
interfaces are agnostic to the concrete (parallel) implementation
of the corresponding objects in the PDE solver, external solvers
are often integrated by creating wrapper classes directly corresponding
to the solvers data structures. However, when the solver is executed
in an MPI distributed context, these wrapper classes will then only
correspond to the rank-local data of a distributed |VectorArray| or
|Operator|.

To facilitate the integration of MPI parallel solvers, we have added
MPI helper classes `[#163] <https://github.com/pymor/pymor/pull/163>`_
in :mod:`pymor.vectorarrays.mpi`, :mod:`pymor.operators.mpi`
and :mod:`pymor.discretizations.mpi` that allow an automatic
wrapping of existing sequential bindings for MPI distributed use.
These wrapper classes are based on a simple event loop provided
by :mod:`pymor.tools.mpi`, which is used in the interface methods of
the wrapper classes to dispatch into MPI distributed execution
of the corresponding methods on the underlying MPI distributed
objects.

The resulting objects can be used on MPI rank 0 (including interactive
Python sessions) without any further changes to pyMOR or the user code.
For an example, see :meth:`pymordemos.thermalblock_simple.discretize_fenics`.


New reduction algorithms
~~~~~~~~~~~~~~~~~~~~~~~~
- :meth:`~pymor.algorithms.adaptivegreedy.adaptive_greedy` uses adaptive
  parameter training set refinement according to [HDO11]_ to prevent
  overfitting of the reduced model to the training set `[#213] <https://github.com/pymor/pymor/pull/213>`_.

- :meth:`~pymor.reductors.parabolic.reduce_parabolic` reduces linear parabolic
  problems using :meth:`~pymor.reductors.basic.reduce_generic_rb` and
  assembles an error estimator similar to [GP05]_, [HO08]_.
  The :mod:`~pymordemos.parabolic_mor` demo contains a simple sample
  application using this reductor `[#190] <https://github.com/pymor/pymor/issues/190>`_.

- The :meth:`~pymor.algorithms.image.estimate_image` and
  :meth:`~pymor.algorithms.image.estimate_image_hierarchical` algorithms
  can be used to find an as small as possible space in which the images of
  a given list of operators for a given source space are contained for all
  possible parameters `mu`. For possible applications, see
  :meth:`~pymor.reductors.residual.reduce_residual` which now uses
  :meth:`~pymor.algorithms.image.estimate_image_hierarchical` for
  Petrov-Galerkin projection of the residual operator `[#223] <https://github.com/pymor/pymor/pull/223>`_.


Copy-on-write semantics for |VectorArrays|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.copy` method
of the |VectorArray| interface is now assumed to have copy-on-write
semantics. I.e., the returned |VectorArray| will contain a reference to the same
data as the original array, and the actual data will only be copied when one of
the arrays is changed. Both |NumpyVectorArray| and |ListVectorArray| have been
updated accordingly `[#55] <https://github.com/pymor/pymor/issues/55>`_.
As a main benefit of this approach, |immutable| objects having a |VectorArray| as
an attribute now can safely create copies of the passed |VectorArrays| (to ensure
the immutability of their state) without having to worry about unnecessarily
increased memory consumption.


Improvements to pyMOR's discretizaion tookit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- An unstructured triangular |Grid| is now provided by :class:`~pymor.grids.unstructured.UnstructuredTriangleGrid`.
  Such a |Grid| can be obtained using the :meth:`~pymor.domaindiscretizers.gmsh.discretize_gmsh`
  method, which can parse `Gmsh <http://gmsh.info/>`_ output files. Moreover, this
  method can generate `Gmsh` input files to create unstructured meshes for
  an arbitrary :class:`~pymor.domaindescriptions.polygonal.PolygonalDomain`
  `[#9] <https://github.com/pymor/pymor/issues/9>`_.

- Basic support for parabolic problems has been added.
  The :meth:`~pymor.discretizers.parabolic.discretize_parabolic_cg` and
  :meth:`~pymor.discretizers.parabolic.discretize_parabolic_fv` methods can
  be used to build continuous finite element or finite volume `Discretizations`
  from a given :class:`pymor.analyticalproblems.parabolic.ParabolicProblem`.
  The :mod:`~pymordemos.parabolic` demo demonstrates the use of these methods
  `[#189] <https://github.com/pymor/pymor/issues/189>`_.

- The :mod:`pymor.discretizers.disk` module contains methods to create stationary and
  instationary affinely decomposed `Discretizations` from matrix data files
  and an `.ini` file defining the given problem.

- :class:`EllipticProblems <pymor.analyticalproblems.elliptic.EllipticProblem>`
  can now also contain advection and reaction terms in addition to the diffusion part.
  :meth:`~pymor.discretizers.cg.discretize_elliptic_cg` has been
  extended accordingly `[#211] <https://github.com/pymor/pymor/pull/211>`_.

- The :mod:`continuous Galerkin <pymor.operators.cg>` module has been extended to
  support Robin boundary conditions `[#110] <https://github.com/pymor/pymor/pull/110>`_.

- :class:`~pymor.functions.bitmap.BitmapFunction` allows to use grayscale
  image data as data |Functions| `[#194] <https://github.com/pymor/pymor/issues/194>`_.

- For the visualization of time-dependent data, the colorbars can now be
  rescaled with each new frame `[#91] <https://github.com/pymor/pymor/pull/91>`_.


Caching improvements
~~~~~~~~~~~~~~~~~~~~
- `state id` generation is now based on deterministic pickling.
  In previous version of pyMOR, the `state id` of |immutable| objects
  was computed from the state ids of the parameters passed to the
  object's `__init__` method. This approach was complicated and error-prone.
  Instead, we now compute the `state id` as a hash of a deterministic serialization
  of the object's state. While this approach is more robust, it is also
  slightly more expensive. However, due to the object's immutability,
  the `state id` only has to be computed once, and state ids are now only
  required for storing results in persistent cache regions (see below).
  Computing such results will usually be much more expensive than the
  `state id` calculation `[#106] <https://github.com/pymor/pymor/issues/106>`_.

- :class:`CacheRegions <pymor.core.cache.CacheRegion>` now have a
  :attr:`~pymor.core.cache.CacheRegion.persistent` attribute indicating
  whether the cache data will be kept between program runs. For persistent
  cache regions the `state id` of the object for which the cached method is
  called has to be computed to obtain a unique persistent id for the given object.
  For non-persistent regions the object's
  `~pymor.core.interfaces.BasicInterface.uid` can be used instead.
  :attr:`pymor.core.cache_regions` now by default contains `'memory'`,
  `'disk'` and `'persistent'` cache regions
  `[#182] <https://github.com/pymor/pymor/pull/182>`_, `[#121] <https://github.com/pymor/pymor/issues/121>`_ .

- |defaults| can now be marked to not affect `state id` computation.
  In previous version of pyMOR, changing any |default| value caused
  a change of the `state id` pyMOR's defaults dictionary, leading to cache
  misses. While this in general is desirable, as, for instance, changed linear
  solver default error tolerances might lead to different solutions for
  the same `Discretization` object, it is clear for many I/O related defaults,
  that these will not affect the outcome of any computation. For these defaults,
  the :meth:`~pymor.core.defaults.defaults` decorator now accepts a `sid_ignore`
  parameter, to exclude these defaults from `state id` computation, preventing
  changes of these defaults causing cache misses `[#81] <https://github.com/pymor/pymor/issues/81>`_.

- As an alternative to using the :meth:`@cached <pymor.core.cache.cached>`
  decorator, :meth:`~pymor.core.cache.CacheableInterface.cached_method_call`
  can be used to cache the results of a function call. This is now used
  in :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
  to enable parsing of the input parameter before it enters the cache key
  calculation `[#231] <https://github.com/pymor/pymor/pull/231>`_.


Additional new features
^^^^^^^^^^^^^^^^^^^^^^^
- :meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse_adjoint` has been added to the |Operator| interface `[#133] <https://github.com/pymor/pymor/issues/133>`_.

- Support for complex values in |NumpyVectorArray| and |NumpyMatrixOperator| `[#131] <https://github.com/pymor/pymor/issues/131>`_.

- New :class:`~pymor.parameters.functionals.ProductParameterFunctional`.
    This |ParameterFunctional| represents the product of a given list of
    |ParameterFunctionals|.

- New :class:`~pymor.operators.constructions.SelectionOperator` `[#105] <https://github.com/pymor/pymor/pull/105>`_.
    This |Operator| represents one |Operator| of a given list of |Operators|,
    depending on the evaluation of a provided |ParameterFunctional|,

- New block matrix operators `[#215] <https://github.com/pymor/pymor/pull/215>`_.
    :class:`~pymor.operators.block.BlockOperator` and
    :class:`~pymor.operators.block.BlockDiagonalOperator` represent block
    matrices of |Operators| which can be applied to appropriately shaped
    :class:`BlockVectorArrays <pymor.vectorarrays.block.BlockVectorArray>`.

- `from_file` factory method for |NumpyVectorArray| and |NumpyMatrixOperator| `[#118] <https://github.com/pymor/pymor/issues/118>`_.
    :meth:`NumpyVectorArray.from_file <pymor.vectorarrays.numpy.NumpyVectorArray.from_file>` and
    :meth:`NumpyMatrixOperator.from_file <pymor.operators.numpy.NumpyMatrixOperator.from_file>`
    can be used to construct such objects from data files of various formats
    (MATLAB, matrix market, NumPy data files, text).

- |ListVectorArray|-based |NumpyMatrixOperator| `[#164] <https://github.com/pymor/pymor/pull/164>`_.
    The :mod:`~pymor.playground` now contains
    :class:`~pymor.playground.operators.numpy.NumpyListVectorArrayMatrixOperator`
    which can apply |NumPy|/|SciPy| matrices to a |ListVectorArray|.
    This |Operator| is mainly intended for performance testing purposes.
    The :mod:`~pymordemos.thermalblock` demo now has an option
    `--list-vector-array` for using this operator instead of |NumpyMatrixOperator|.

- Log indentation support `[#230] <https://github.com/pymor/pymor/pull/230>`_.
    pyMOR's log output can now be indented via the `logger.block(msg)`
    context manger to reflect the hierarchy of subalgorithms.

- Additional `INFO2` and `INFO3` log levels `[#212] <https://github.com/pymor/pymor/pull/212>`_.
    :mod:`Loggers <pymor.core.logger>` now have additional `info2`
    and `info3` methods to highlight important information (which does
    fall in the 'warning' category).

- Default implementation of :meth:`~pymor.operators.interfaces.OperatorInterface.as_vector` for functionals `[#107] <https://github.com/pymor/pymor/issues/107>`_.
    :meth:`OperatorBase.as_vector <pymor.operators.basic.OperatorBase>` now
    contains a default implementation for functionals by calling
    :meth:`~pymor.operators.interfaces.OperatorInterface.apply_adjoint`.

- `pycontracts` has been removed as a dependency of pyMOR `[#127] <https://github.com/pymor/pymor/pull/127>`_.

- Test coverage has been raised to 80 percent.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |VectorArray| implementations have been moved to the :mod:`pymor.vectorarrays` sub-package `[#89] <https://github.com/pymor/pymor/issues/89>`_.

- The `dot` method of the |VectorArray| interface has been split into :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dot` and :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.pairwise_dot` `[#76] <https://github.com/pymor/pymor/issues/76>`_.
    The `pairwise` parameter of :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.dot`
    has been removed, always assuming `pairwise == False`. The method
    :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.pairwise_dot`
    corresponds to the `pairwise == True` case. Similarly the `pariwise` parameter
    of the :meth:`~pymor.operators.interfaces.OperatorInterface.apply2` method
    of the |Operator| interface has been removed and a
    :meth:`~pymor.operators.interfaces.OperatorInterface.pairwise_apply2` method
    has been added.

- `almost_equal` has been removed from the |VectorArray| interface `[#143] <https://github.com/pymor/pymor/issues/143>`_.
    As a replacement, the new method :meth:`pymor.algorithms.basic.almost_equal`
    can be used to compare |VectorArrays| for almost equality by the norm
    of their difference.

- `lincomb` has been removed from the |Operator| interface `[#83] <https://github.com/pymor/pymor/issues/83>`_.
    Instead, a |LincombOperator| should be directly instantiated.

- Removal of the `options` parameter of :meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse` in favor of :attr:`~pymor.operators.interfaces.OperatorInterface.solver_options` attribute `[#122] <https://github.com/pymor/pymor/issues/122>`_.
    The `options` parameter of :meth:`OperatorInterface.apply_inverse <pymor.operators.interfaces.OperatorInterface.apply_inverse>`
    has been replaced by the :attr:`~pymor.operators.interfaces.OperatorInterface.solver_options`
    attribute. This attribute controls which fixed (linear) solver options are
    used when :meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse` is
    called. See `here <https://github.com/pymor/pymor/pull/184>`__ for more details.

- Renaming of reductors for coercive problems `[#224] <https://github.com/pymor/pymor/issues/224>`_.
    :meth:`pymor.reductors.linear.reduce_stationary_affine_linear` and
    :meth:`pymor.reductors.stationary.reduce_stationary_coercive` have been
    renamed to :meth:`pymor.reductors.coercive.reduce_coercive` and
    :meth:`pymor.reductors.coercive.reduce_coercive_simple`. The old names
    are deprecated and will be removed in pyMOR 0.5.

- Non-parametric objects have now `~pymor.parameters.base.Parametric.parameter_type` `{}` instead of `None` `[#84] <https://github.com/pymor/pymor/issues/84>`_.

- Sampling methods of |ParameterSpaces| now return iterables instead of iterators `[#108] <https://github.com/pymor/pymor/issues/108>`_.

- Caching of :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve` is now disabled by default `[#178] <https://github.com/pymor/pymor/issues/178>`_.
    Caching of :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    must now be explicitly enabled by using
    :meth:`pymor.core.cache.CacheableInterface.enable_caching`.

- The default value for `extension_algorithm` parameter of :meth:`~pymor.algorithms.greedy.greedy` has been removed `[#82] <https://github.com/pymor/pymor/issues/82>`_.

- Changes to :meth:`~pymor.algorithms.ei.ei_greedy` `[#159] <https://github.com/pymor/pymor/issues/159>`_, `[#160] <https://github.com/pymor/pymor/issues/160>`_.
    The default for the `projection` parameter has been changed from `'orthogonal'`
    to `'ei'` to let the default algorithm agree with literature. In
    addition a `copy` parameter with default `True` has been added.
    When `copy` is `True`, the input data is copied before executing
    the algorithm, ensuring, that the original |VectorArray| is left
    unchanged. When possible, `copy` should be set to `False` in order
    to reduce memory consumption.

- The `copy` parameter of :meth:`pymor.algorithms.gram_schmidt.gram_schmidt` now defaults to `True` `[#123] <https://github.com/pymor/pymor/issues/123>`_.

- `with_` has been moved from `BasicInterface` to `ImmutableInterface` `[#154] <https://github.com/pymor/pymor/issues/154>`_.

- `BasicInterface.add_attributes` has been removed `[#158] <https://github.com/pymor/pymor/issues/158>`_.

- Auto-generated names no longer contain the :attr:`~pymor.core.interfaces.BasicInterface.uid` `[#198] <https://github.com/pymor/pymor/issues/198>`_.
    The auto-generated `~pymor.core.interfaces.BasicInterface.name`
    of pyMOR objects no longer contains their
    `~pymor.core.interfaces.BasicInterface.uid`. Instead, the name
    is now simply set to the class name.

- Python fallbacks to Cython functions have been removed `[#145] <https://github.com/pymor/pymor/issues/145>`_.
    In order to use pyMOR's discretization toolkit, building of the
    :mod:`~pymor.grids._unstructured`, :mod:`~pymor.tools.inplace`,
    :mod:`~pymor.tools.relations` Cython extension modules is now
    required.



Further improvements
^^^^^^^^^^^^^^^^^^^^

- `[#78] update apply_inverse signature <https://github.com/pymor/pymor/issues/78>`_
- `[#115] [algorithms.gram_schmidt] silence numpy warning <https://github.com/pymor/pymor/issues/115>`_
- `[#144] L2ProductP1 uses wrong quadrature rule in 1D case <https://github.com/pymor/pymor/issues/144>`_
- `[#147] Debian doc packages have weird title <https://github.com/pymor/pymor/issues/147>`_
- `[#151] add tests for 'almost_equal' using different norms <https://github.com/pymor/pymor/issues/151>`_
- `[#156] Let thermal block demo use error estimator by default <https://github.com/pymor/pymor/issues/156>`_
- `[#195] Add more tests / fixtures for operators in pymor.operators.constructions <https://github.com/pymor/pymor/issues/195>`_
- `[#197] possible problem in caching <https://github.com/pymor/pymor/issues/197>`_
- `[#207] No useful error message in case PySide.QtOpenGL cannot be imported <https://github.com/pymor/pymor/issues/207>`_
- `[#209] Allow 'pip install pymor' to work even when numpy/scipy are not installed yet <https://github.com/pymor/pymor/issues/209>`_
- `[#219] add minimum versions for dependencies <https://github.com/pymor/pymor/issues/219>`_
- `[#228] merge fixes in python3 branch back to master <https://github.com/pymor/pymor/issues/228>`_
- `[#269] Provide a helpful error message when cython modules are missing <https://github.com/pymor/pymor/issues/269>`_
- `[#276] Infinite recursion in apply for IdentityOperator * scalar <https://github.com/pymor/pymor/issues/276>`_





pyMOR 0.3 (March 2, 2015)
-------------------------

- Introduction of the vector space concept for even simpler
  integration with external solvers.

- Addition of a generic Newton algorithm.

- Support for Jacobian evaluation of empirically interpolated operators.

- Greatly improved performance of the EI-Greedy algorithm. Addition of
  the DEIM algorithm.

- A new algorithm for residual operator projection and a new,
  numerically stable a posteriori error estimator for stationary coercive
  problems based on this algorithm. (cf. A. Buhr, C. Engwer, M. Ohlberger,
  S. Rave, 'A numerically stable a posteriori error estimator for reduced
  basis approximations of elliptic equations', proceedings of WCCM 2014,
  Barcelona, 2014.)

- A new, easy to use mechanism for setting and accessing default values.

- Serialization via the pickle module is now possible for each class in
  pyMOR. (See the new 'analyze_pickle' demo.)

- Addition of generic iterative linear solvers which can be used in
  conjunction with any operator satisfying pyMOR's operator interface.
  Support for least squares solvers and PyAMG (http://www.pyamg.org/).

- An improved SQLite-based cache backend.

- Improvements to the built-in discretizations: support for bilinear
  finite elements and addition of a finite volume diffusion operator.

- Test coverage has been raised from 46% to 75%.

Over 500 single commits have entered this release. A full list of
all changes can be obtained under the following address:
https://github.com/pymor/pymor/compare/0.2.2...0.3.0
