---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:load: myst_code_init.py
:tags: [remove-cell]


```

# Tutorial: Parameters

Model order reduction is mostly concerned with models that depend on parameters,
such as material coefficients, the position of a heat source or the strength of
an inflow.
pyMOR's parameter handling consists of a few small building blocks:

- {{ Parameters }} describe *which* parameters an object depends on, i.e., their names
  and dimensions.
- {{ Mu }} objects store concrete {{ parameter_values }}.
- {{ ParametricObjects }} like {{ Functions }}, {{ Operators }} or {{ Models }} know which
  {{ Parameters }} they depend on and accept {{ parameter_values }} in their methods.
- {{ ParameterFunctionals }} map {{ parameter_values }} to numbers and are used to express
  parameter-separable decompositions.
- {{ ParameterSpaces }} equip {{ Parameters }} with ranges, so that {{ parameter_values }}
  can be sampled.

At the end of this tutorial, we will also see how pyMOR handles time as a parameter
and how {{ parameter_values }} can vary in time.

## Parameters: names and dimensions

Each parameter in pyMOR has a name and a fixed dimension, i.e., the number of its scalar
components.
In particular, there are no scalar parameters in pyMOR:
a scalar parameter is simply a parameter of dimension one.
A collection of parameters is described by a {{ Parameters }} object, which is an
immutable dictionary mapping parameter names to their dimensions:

```{code-cell} ipython3
from pymor.basic import *
import numpy as np

parameters = Parameters({'diffusion': 2, 'source': 1})
parameters
```

{{ Parameters }} can also be created using keyword arguments.
Note that the parameters are always sorted alphabetically by their names:

```{code-cell} ipython3
Parameters(source=1, diffusion=2)
```

The total number of scalar components is given by
{attr}`~pymor.parameters.base.Parameters.dim`:

```{code-cell} ipython3
parameters.dim
```

{{ Parameters }} can be merged using `|` and removed using `-`:

```{code-cell} ipython3
print(parameters | Parameters(velocity=3))
print(parameters - Parameters(source=1))
```

In practice, you will rarely create {{ Parameters }} yourself.
Instead, you will find them as the
{attr}`~pymor.parameters.base.ParametricObject.parameters` attribute of a given
{{ ParametricObject }} (see below).

## Parameter values: Mu

Concrete {{ parameter_values }} are stored in {{ Mu }} objects.
A {{ Mu }} is an immutable dictionary mapping parameter names to one-dimensional
{{ NumPy_arrays }}.
The recommended way to create a {{ Mu }} is the
{meth}`~pymor.parameters.base.Parameters.parse` method of {{ Parameters }}.
It checks that the given values match the {{ Parameters }} and converts them into
the correct format:

```{code-cell} ipython3
mu = parameters.parse({'diffusion': [1., 2.], 'source': 3.})
mu
```

Note how the value `3.` for `'source'` has been converted to an array of length one.
Instead of a dictionary, we can also pass a list or a {{ NumPy_array }} with all values.
In this case, the values are assigned to the parameters in alphabetical order:

```{code-cell} ipython3
parameters.parse([1., 2., 3.])
```

If there is only a single parameter, a single number is enough:

```{code-cell} ipython3
Parameters(source=1).parse(3.)
```

If the given values do not match the {{ Parameters }},
{meth}`~pymor.parameters.base.Parameters.parse` raises an error:

```{code-cell} ipython3
try:
    parameters.parse({'diffusion': [1., 2., 3.], 'source': 3.})
except ValueError as e:
    print(e)
```

The values stored in a {{ Mu }} are accessed like in a dictionary:

```{code-cell} ipython3
print(mu['diffusion'])
print(mu['source'][0])
print(list(mu.keys()))
```

As {{ Mu }} objects are immutable, the stored arrays cannot be modified:

```{code-cell} ipython3
try:
    mu['source'][0] = 42.
except ValueError as e:
    print(e)
```

To get a {{ Mu }} where some of the values are changed, use
{meth}`~pymor.parameters.base.Mu.with_`:

```{code-cell} ipython3
mu.with_(source=42.)
```

{meth}`~pymor.parameters.base.Mu.to_numpy` concatenates all values into a single array
using alphabetical ordering.
This is the inverse of parsing a list of values:

```{code-cell} ipython3
mu.to_numpy()
```

Since {{ Mu }} objects are immutable, they can be compared and hashed.
pyMOR uses this, for instance, to {mod}`cache <pymor.core.cache>` solutions of {{ Models }}
for given {{ parameter_values }}:

```{code-cell} ipython3
mu == parameters.parse([1., 2., 3.])
```

It is also possible to instantiate {{ Mu }} directly, e.g., `Mu(diffusion=[1., 2.], source=3.)`.
However, no checks against any {{ Parameters }} are performed in this case,
so {meth}`~pymor.parameters.base.Parameters.parse` is usually the better choice.

To check whether a {{ Mu }} contains values for given {{ Parameters }}, use
{meth}`~pymor.parameters.base.Parameters.is_compatible`.
A {{ Mu }} is compatible if it contains values with the correct dimension for each
of the parameters.
It may contain values for additional parameters, though:

```{code-cell} ipython3
print(Parameters(source=1).is_compatible(mu))
print(Parameters(velocity=3).is_compatible(mu))
```

## Parametric objects

All objects in pyMOR that may depend on parameters derive from {{ ParametricObject }}.
In particular, this includes all {{ Functions }}, {{ Operators }} and {{ Models }}.
The {{ Parameters }} the object depends on are available via its
{attr}`~pymor.parameters.base.ParametricObject.parameters` attribute.
For instance, here is an {{ ExpressionFunction }} depending on a two-dimensional
parameter `'c'`:

```{code-cell} ipython3
f = ExpressionFunction('c[0] * x[0]**2 + c[1]', 1, parameters={'c': 2})
f.parameters
```

The {{ parametric }} attribute tells us whether an object depends on any parameter at all:

```{code-cell} ipython3
print(f.parametric)
print(ConstantFunction(1., 1).parametric)
```

To evaluate the function, we need to pass {{ parameter_values }}:

```{code-cell} ipython3
x = np.linspace(0, 1, 3).reshape((-1, 1))
f(x, mu=f.parameters.parse([2., 1.]))
```

```{note}
Methods like {meth}`Function.evaluate <pymor.analyticalproblems.functions.Function.evaluate>`
or {meth}`Operator.apply <pymor.operators.interface.Operator.apply>`
only accept {{ Mu }} instances.
Only the methods of {{ Models }} like {meth}`~pymor.models.interface.Model.solve` or
{meth}`~pymor.models.interface.Model.output` call
{meth}`~pymor.parameters.base.Parameters.parse` for you (see below).
```

```{code-cell} ipython3
try:
    f(x, mu=[2., 1.])
except TypeError as e:
    print(e)
```

### Parameter functionals

{{ ParameterFunctionals }} map {{ parameter_values }} to scalars.
pyMOR comes with several of them, most importantly:

- {class}`~pymor.parameters.functionals.ProjectionParameterFunctional`, which returns one
  component of a given parameter,
- {{ ExpressionParameterFunctional }}, which evaluates a given expression string,
- {class}`~pymor.parameters.functionals.GenericParameterFunctional`, which wraps an arbitrary
  Python function.

```{code-cell} ipython3
theta_1 = ProjectionParameterFunctional('diffusion')
theta_2 = ExpressionParameterFunctional('exp(-c[0]) * c[1]', {'c': 2})
print(theta_1.parameters)
print(theta_2.parameters)
```

{meth}`~pymor.parameters.base.Parameters.of` computes the {{ Parameters }} of a collection
of {{ ParametricObjects }}, so we can create a {{ Mu }} for both functionals at once:

```{code-cell} ipython3
mu = Parameters.of(theta_1, theta_2).parse({'c': [0., 2.], 'diffusion': 5.})
print(theta_1(mu))
print(theta_2(mu))
```

{{ ParameterFunctionals }} support arithmetic operations.
The result depends on the {{ Parameters }} of all involved functionals:

```{code-cell} ipython3
theta = 2 * theta_1 + theta_2 - 1
print(theta.parameters)
print(theta(mu))
```

Partial derivatives w.r.t. a parameter component are obtained using
{meth}`~pymor.parameters.functionals.ParameterFunctional.d_mu`:

```{code-cell} ipython3
theta_1.d_mu('diffusion')(mu)
```

For {{ ExpressionParameterFunctionals }}, the derivatives have to be provided via
the `derivative_expressions` argument.
See {doc}`tutorial_optimization` for an application.

### How parameters are inherited

Usually, the {{ Parameters }} of a {{ ParametricObject }} are determined automatically:
they are the union of the {{ Parameters }} of all objects passed to its `__init__`
method.
This is particularly useful for parameter-separable (affine) decompositions.
Consider a diffusion coefficient on {math}`[0, 1]`, which is fixed in the left half of the
domain and given by a parameter in the right half:

```{math}
\sigma(x, \mu) = 1 \cdot \chi_{[0, 0.5]}(x) + \mu_{\text{diffusion}} \cdot \chi_{(0.5, 1]}(x).
```

We represent it as a {{ LincombFunction }}:

```{code-cell} ipython3
diffusion = LincombFunction(
    [ExpressionFunction('(x[0] <= 0.5) * 1.', 1),
     ExpressionFunction('(x[0] > 0.5) * 1.', 1)],
    [1., ProjectionParameterFunctional('diffusion')]
)
diffusion.parameters
```

The same works for {{ Operators }}.
Here, we form a {{ LincombOperator }} by multiplying an {{ Operator }} with a
{{ ParameterFunctional }}:

```{code-cell} ipython3
A_0 = NumpyMatrixOperator(np.eye(3))
A_1 = NumpyMatrixOperator(np.diag([1., 2., 3.]))
A = A_0 + ExpressionParameterFunctional('k[0]**2', {'k': 1}) * A_1
print(A)
print(A.assemble(A.parameters.parse(2.)).matrix)
```

{meth}`~pymor.parameters.base.Parameters.of` also checks that parameters with the same
name have the same dimension:

```{code-cell} ipython3
print(Parameters.of(diffusion, A, f))
try:
    Parameters.of(f, ExpressionFunction('c[0]', 1, parameters={'c': 1}))
except AssertionError as e:
    print(e)
```

### Parametric models

pyMOR's discretizers preserve this structure.
Let us use the diffusion coefficient from above for a one-dimensional stationary heat
equation, where the position of the heat source is another parameter.
As output, we compute the integral of the solution over the domain.

```{code-cell} ipython3
problem = StationaryProblem(
    domain=LineDomain([0., 1.]),
    diffusion=diffusion,
    rhs=ExpressionFunction('exp(-((x[0] - position[0]) / 0.05)**2)', 1,
                           parameters={'position': 1}),
    outputs=[('l2', ConstantFunction(1., 1))],
    parameter_ranges={'diffusion': (0.1, 10.), 'position': (0.1, 0.9)},
)
fom, _ = discretize_stationary_cg(problem, diameter=1/100)
fom.parameters
```

Each {{ Operator }} of the {{ Model }} only depends on the parameters it actually needs:

```{code-cell} ipython3
print(fom.operator.parameters)
print(fom.rhs.parameters)
```

As already mentioned, the methods of a {{ Model }} accept everything that
{meth}`~pymor.parameters.base.Parameters.parse` accepts:

```{code-cell} ipython3
print(fom.output({'diffusion': 1., 'position': 0.5}))
print(fom.output([1., 0.5]))
```

```{code-cell} ipython3
fom.visualize((fom.solve([1., 0.3]), fom.solve([10., 0.3])),
              legend=('diffusion = 1', 'diffusion = 10'))
```

## Implementing parametric objects

When you implement your own {{ Function }}, {{ Operator }} or {{ Model }}, there are three
ways to control its {{ Parameters }}:

1. Do nothing.
   The {{ Parameters }} are computed from all `__init__` arguments, which are expected to be
   stored as attributes of the same name.
   Lists, tuples and dicts of {{ ParametricObjects }} are traversed as well.
2. Set {attr}`~pymor.parameters.base.ParametricObject.parameters_own` in `__init__`
   to add parameters the object itself depends on.
3. Set {attr}`~pymor.parameters.base.ParametricObject.parameters_internal` in `__init__`
   to remove parameters which are fixed by the object.

As an example, consider a {{ Function }} that multiplies another {{ Function }} by an
additional scalar parameter:

```{code-cell} ipython3
from pymor.analyticalproblems.functions import Function

class ScaledFunction(Function):

    def __init__(self, function, scaling_parameter):
        self.function = function
        self.scaling_parameter = scaling_parameter
        self.dim_domain = function.dim_domain
        self.shape_range = function.shape_range
        self.parameters_own = {scaling_parameter: 1}

    def evaluate(self, x, mu=None):
        assert self.parameters.assert_compatible(mu)
        return mu[self.scaling_parameter][0] * self.function(x, mu=mu)
```

The {{ Parameters }} of `ScaledFunction` consist of the parameters inherited from
`function` and the parameter introduced by the object itself:

```{code-cell} ipython3
g = ScaledFunction(f, 'scale')
print(g.parameters)
print(g.parameters_own)
print(g.parameters_inherited)
g(x, mu=g.parameters.parse({'c': [2., 1.], 'scale': 10.}))
```

In `evaluate`, we used {meth}`~pymor.parameters.base.Parameters.assert_compatible`
to check the given {{ parameter_values }}.
It raises an error with a helpful message when a value is missing:

```{code-cell} ipython3
try:
    g(x, mu=f.parameters.parse([2., 1.]))
except AssertionError as e:
    print(e)
```

Wrapping the call in an `assert` statement ensures that the check is skipped when Python
is run with the `-O` flag.

```{note}
In practice, you would not need `ScaledFunction`, as the same can be achieved with
`LincombFunction([f], [ProjectionParameterFunctional('scale')])`.
```

Fixing parameters is done, e.g., by
{class}`~pymor.operators.constructions.FixedParameterOperator`.
The parameters of the wrapped {{ Operator }} become internal and the resulting
{{ Operator }} is no longer parametric:

```{code-cell} ipython3
from pymor.operators.constructions import FixedParameterOperator

A_fixed = FixedParameterOperator(A, A.parameters.parse(2.))
print(A_fixed.parameters)
print(A_fixed.parameters_internal)
print(A_fixed.parametric)
```

{{ Models }} use the same mechanism for their `input` (see
{ref}`below <time-dependent-parameter-values>`):

```{code-cell} ipython3
fom.parameters_internal
```

## Parameter spaces

To build a reduced-order model, we usually need to solve the full-order model for many
{{ parameter_values }}.
A {{ ParameterSpace }} defines the ranges of the parameters and is used to sample
{{ parameter_values }}.
We can create one using the {meth}`~pymor.parameters.base.Parameters.space` method of
{{ Parameters }}:

```{code-cell} ipython3
space = fom.parameters.space({'diffusion': (0.1, 10.), 'position': (0.1, 0.9)})
space
```

If all parameters should have the same range, we can simply write
`fom.parameters.space(0.1, 10.)`.
Since we have specified `parameter_ranges` for our analytical problem, we could also
have used `problem.parameter_space`.

{meth}`~pymor.parameters.base.ParameterSpace.sample_uniformly` returns a tensor grid of
{{ parameter_values }}.
The number of samples can be specified per parameter.
For a parameter with more than one component, each component is sampled with the given
number of points, so `n` points for a parameter of dimension `d` result in `n**d` samples.

```{code-cell} ipython3
training_set = space.sample_uniformly({'diffusion': 3, 'position': 20})
len(training_set)
```

The samples are ordered like nested loops over the parameters in alphabetical order.
We use this to plot the output over the position of the heat source:

```{code-cell} ipython3
import matplotlib.pyplot as plt

outputs = np.array([fom.output(mu)[0, 0] for mu in training_set]).reshape((3, 20))
positions = [mu['position'][0] for mu in training_set[:20]]
fig, ax = plt.subplots()
for mu, y in zip(training_set[::20], outputs):
    ax.plot(positions, y, label=f"diffusion = {mu['diffusion'][0]}")
ax.set_xlabel('position')
ax.set_ylabel('output')
ax.legend()
plt.show()
```

{meth}`~pymor.parameters.base.ParameterSpace.sample_randomly` draws random samples using
pyMOR's global {{ RNG }}.
To obtain reproducible samples, use {func}`~pymor.tools.random.new_rng`:

```{code-cell} ipython3
with new_rng(0):
    print(space.sample_randomly(2))
with new_rng(0):
    print(space.sample_randomly(2))
```

Note that the outputs for `diffusion = 5.05` and `diffusion = 10.0` are almost the same,
while there is no sample between `0.1` and `5.05`.
Since the diffusion coefficient varies over two orders of magnitude, it is better to
sample it logarithmically using
{meth}`~pymor.parameters.base.ParameterSpace.sample_logarithmic_uniformly` or
{meth}`~pymor.parameters.base.ParameterSpace.sample_logarithmic_randomly`:

```{code-cell} ipython3
Parameters(diffusion=1).space(0.1, 10.).sample_logarithmic_uniformly(3)
```

{meth}`~pymor.parameters.base.ParameterSpace.contains` checks if given
{{ parameter_values }} lie in the space, and
{meth}`~pymor.parameters.base.ParameterSpace.clip` projects them onto it:

```{code-cell} ipython3
print(space.contains([100., 0.5]))
print(space.clip([100., 0.5]))
```

Finally, additional constraints can be specified by a function returning `True` for
admissible {{ parameter_values }}.
Here, we restrict a two-dimensional parameter to a triangle:

```{code-cell} ipython3
triangle = Parameters(p=2).space(0., 1., constraints=lambda mu: mu['p'].sum() <= 1.)
samples = np.array([mu['p'] for mu in triangle.sample_randomly(500)])

fig, ax = plt.subplots()
ax.scatter(samples[:, 0], samples[:, 1], s=4)
ax.set_aspect('equal')
plt.show()
```

## Time-dependent parameters

For instationary problems, there are two ways in which time can come into play:
data functions may explicitly depend on time, and the {{ parameter_values }} themselves
may vary in time.

### The time parameter t

pyMOR treats time as a special parameter named `'t'` that always has dimension one.
For instance, a heat source that is periodically switched on and off can be modeled as

```{code-cell} ipython3
source = (ExpressionFunction('exp(-((x[0] - 0.5) / 0.05)**2)', 1)
          * ExpressionParameterFunctional('sin(2*pi*t[0])**2', {'t': 1}))
source.parameters
```

We use it for an {{ InstationaryProblem }}, where we now measure the integral of the
solution over the right half of the domain:

```{code-cell} ipython3
inst_problem = InstationaryProblem(
    StationaryProblem(
        domain=LineDomain([0., 1.]),
        diffusion=diffusion,
        rhs=source,
        outputs=[('l2', ExpressionFunction('(x[0] > 0.5) * 1.', 1))],
    ),
    initial_data=ConstantFunction(0., 1),
    T=2.,
)
inst_fom, _ = discretize_instationary_cg(inst_problem, diameter=1/100, nt=200)
print(inst_problem.parameters)
print(inst_fom.parameters)
```

While the analytical problem depends on `'t'`, the {{ InstationaryModel }} does not:
the model's time-stepper provides the value for `'t'` while solving,
so `'t'` is an internal parameter of the model:

```{code-cell} ipython3
print(inst_fom.parameters_internal)
print(inst_fom.rhs.parameters)
```

To evaluate the time-dependent {{ Operators }}, the time-stepper calls
{meth}`~pymor.parameters.base.Mu.at_time`,
which adds the current time to the {{ parameter_values }}:

```{code-cell} ipython3
mu = inst_fom.parameters.parse(1.)
mu.at_time(0.25)
```

(time-dependent-parameter-values)=

### Time-dependent parameter values

Sometimes, the parameters themselves vary in time, e.g., when they describe a control.
In pyMOR, this does not require any changes to the {{ Model }}.
Instead, a function of time can be passed as the value for a parameter.
Strings are interpreted as expressions in `t` and converted to an {{ ExpressionFunction }}:

```{code-cell} ipython3
mu_t = inst_fom.parameters.parse({'diffusion': '1 - 0.45 * t[0]'})
print(mu_t)
```

Values for vector-valued parameters can be given as a list mixing numbers and expressions,
e.g., `[1., 'sin(t[0])']`.
Time-dependent values are stored separately in
{attr}`~pymor.parameters.base.Mu.time_dependent_values` and are not visible via
the dictionary interface of {{ Mu }}:

```{code-cell} ipython3
print('diffusion' in mu_t)
print(list(mu_t.keys()))
print(list(mu_t.time_dependent_values.keys()))
```

The reason for this is that most code in pyMOR does not know anything about time.
A function of time passed to, e.g., the `apply` method of a stationary {{ Operator }}
would lead to wrong results or obscure errors.
Thus, {meth}`~pymor.parameters.base.Parameters.is_compatible` ignores time-dependent values
unless `allow_time_dependent=True` is passed:

```{code-cell} ipython3
print(inst_fom.parameters.is_compatible(mu_t))
print(inst_fom.parameters.is_compatible(mu_t, allow_time_dependent=True))
```

Time-aware code like pyMOR's time-steppers uses {meth}`~pymor.parameters.base.Mu.at_time`
to evaluate the time-dependent values at the current time:

```{code-cell} ipython3
mu_t.at_time(1.)
```

Let us compare the output of our model for constant and time-dependent diffusion.
At the beginning, the time-dependent diffusion is close to `1` and at the end close to
`0.1`:

```{code-cell} ipython3
times = np.linspace(0., inst_fom.T, 201)
fig, ax = plt.subplots()
for value in [1., 0.1, '1 - 0.45 * t[0]']:
    ax.plot(times, inst_fom.output({'diffusion': value})[0], label=f'diffusion = {value}')
ax.set_xlabel('t')
ax.set_ylabel('output')
ax.legend(loc='upper left')
plt.show()
```

Passing time-dependent values to a {{ StationaryModel }} raises an error:

```{code-cell} ipython3
try:
    fom.solve({'diffusion': '1 + t[0]', 'position': 0.5})
except AssertionError as e:
    print(e)
```

The `input` of {{ Models }} like {{ LTIModel }} is handled in the same way.
When calling, e.g., `m.output(mu, input='sin(t[0])')`, the input is parsed like a
parameter value and added to `mu` as a value for the internal parameter `'input'`.
See {doc}`tutorial_lti_systems` for examples.

Download the code:
{download}`tutorial_parameters.md`
{nb-download}`tutorial_parameters.ipynb`
