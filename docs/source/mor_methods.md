---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{try_on_binder}
```

```{code-cell} ipython3
:load: myst_code_init.py
:tags: [remove-cell]


```

# Available MOR methods

Here we give an overview over (most of) the available MOR methods implemented in pyMOR.
We provide short code snippets that show how to use these methods with pyMOR.
For more in-depth explanations we refer to the {doc}`tutorials`.

## Data approximation

### POD

```{code-cell} ipython3
:tags: [remove-output]

# generate some data to approximate
from pymor.models.examples import thermal_block_example
fom = thermal_block_example(diameter=1/10)
U = fom.solution_space.empty()
for mu in fom.parameters.space(0.1, 1).sample_randomly(10):
    U.append(fom.solve(mu))

# return first 3 POD modes and singular values
from pymor.algorithms.pod import pod
modes, singular_values = pod(U, modes=3)

# return modes with singular value larger than 1e-3
modes, _  = pod(U, atol=1e-3)

# return right-singular vectors
modes, _, coeffs = pod(U, return_reduced_coefficients=True)

# use slower but more accurate algorithm
# (default algorithm is only accurate up to half machine precision)
modes, _ = pod(U, method='qr_svd')
```

## Parametric MOR

### Available models

```{admonition} {{InstationaryModel}}
:class: dropdown

**Description**
Generic class for models of instationary problems. This class describes instationary problems given by the equations

$$
M \cdot \partial_t u(t, \mu) + L(u(\mu), t, \mu) = F(t, \mu) \\
u(0, \mu) = u_0(\mu)
$$

For $t\in[0, T]$, L being a (possibly non-linear) time-dependent {{Operator}}, F a time-dependent vector-like {{Operator}}, and $u_0$ the initial data. The mass {{Operator}} M is assumed to be linear.

**Initializing the InstationaryModel Class**

```python
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import ZeroOperator, IdentityOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import VectorOperator
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper

# Define the vector space
vector_space = NumpyVectorSpace(1)

# Mandatory parameters
T = 10.0  # Final time
initial_data = VectorOperator(vector_space.make_array([[3.0]]), name='initial_data')
operator = IdentityOperator(vector_space)       # Operator L
rhs = ZeroOperator(vector_space, vector_space)  # Right-hand side F
time_stepper = ExplicitEulerTimeStepper(nt=10)  # Explicit Euler time-stepper with 10 steps

# Optional parameters
mass = IdentityOperator(vector_space)  # Mass operator M
num_values = 5  # Return 5 solution trajectory vectors
output_functional = IdentityOperator(vector_space)  # Output functional

# Initialize model with mandatory parameters
fom_basic = InstationaryModel(T=T, initial_data=initial_data, operator=operator, rhs=rhs, time_stepper=time_stepper)

# Initialize model with additional parameters
fom_detailed = InstationaryModel(T=T, initial_data=initial_data, operator=operator, rhs=rhs, mass=mass, time_stepper=time_stepper, num_values=num_values, output_functional=output_functional)

print('InstationaryModel with only mandatory parameters: \n{}\n'.format(fom_basic))
print('InstationaryModel with additional parameters: \n{}\n'.format(fom_detailed))
```

```{admonition} {{QuadraticHamiltonianModel}}
:class: dropdown

**Description**
This class describes Hamiltonian systems given by the equations:

$$
\frac{\partial u(t, \mu)}{\partial t} = J H_{\text{op}}(t, \mu) u(t, \mu) + J h(t, \mu)
$$

$$
u(0, \mu) = u_0(\mu)
$$

for $t\in[0,T]$, where $H_{\text{op}}$ is a linear time-dependent {{Operator}}, $J$ is a canonical Poisson matrix, $h$ is a (possibly) time-dependent vector-like {{Operator}}, and $u_0$ the initial data.

**Initializing the QuadraticHamiltonianModel Class**

```python
from pymor.models.symplectic import QuadraticHamiltonianModel
from pymor.operators.constructions import VectorOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper

import numpy as np

# Define the phase space
dim = 4  # Must be even for Hamiltonian systems
phase_space = NumpyVectorSpace(dim)
H_op = NumpyMatrixOperator(np.eye(dim)) # Define the Hamiltonian operator (H_op) as an identity matrix

# Define the initial condition as a VectorArray of length 1
initial_data = phase_space.from_numpy(np.array([[1.0, 0.0, -1.0, 0.5]]))

T = 4.0 # Define the final time
time_stepper = ExplicitEulerTimeStepper(nt=10) # Explicit Euler time-stepper with 10 steps
h = VectorOperator(phase_space.from_numpy(np.array([[0.1, -0.2, 0.3, -0.4]]))) # state-independent Hamiltonian term h

# Initialize the QuadraticHamiltonianModel
fom = QuadraticHamiltonianModel(T=T, initial_data=initial_data, H_op=H_op, time_stepper=time_stepper, h=h)

print('QuadraticHamiltonianModel: \n{}\n'.format(fom))
```

```{admonition} {{StationaryModel}}
:class: dropdown

**Description**
Generic class for models of stationary problems. This class describes discrete problems given by the equation:

$$
L(u(\mu), \mu) = F(\mu)
$$

with a vector-like right-hand side F and a (possibly non-linear) operator L.

**Initializing the StationaryModel Class**

```python
from pymor.models.basic import StationaryModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.constructions import VectorOperator

# Define the vector space
vector_space = NumpyVectorSpace(1)

# Mandatory parameters
operator = IdentityOperator(vector_space)  # Operator L
rhs = VectorOperator(vector_space.make_array([[1.0]]), name="rhs")  # RHS F

# Optional parameters
output_functional = IdentityOperator(vector_space)  # Output functional
products = {"l2": IdentityOperator(vector_space)}  # L2 inner product
output_d_mu_use_adjoint = True  # Use adjoint solution for output gradients

# Initialize model with mandatory parameters
fom_basic = StationaryModel(operator=operator, rhs=rhs)

# Initialize model with additional parameters
fom_detailed = StationaryModel(operator=operator, rhs=rhs, output_functional=output_functional, products=products, output_d_mu_use_adjoint=output_d_mu_use_adjoint)

print('StationaryModel with only mandatory parameters: \n{}\n'.format(fom_basic))
print('StationaryModel with additional parameters: \n{}\n'.format(fom_detailed))
```

Here we consider MOR methods for {{Models}} that depend on one or more {{Parameters}}.

### Reduced Basis method for parameter-separable, linear, coercive models

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example(diameter=1/10)

# FOM is parameter separable, i.e., system operator is a
# linear combination of non-parametric operators with parametric coefficients
print(repr(fom.operator))

# instantiate reductor that builds the ROM given some reduced basis;
# `product` is inner product w.r.t. which MOR error is estimated;
# `coercivity_estimator` needs to return lower bound for the operator's
# coercivity constant (w.r.t. given `product`) for the given parameter values
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)

# note: use SimpleCoerciveRBReductor for faster offline phase but error estimator that
# only is accurate up to half machine precision

# use weak greedy algorithm to train the model
from pymor.algorithms.greedy import rb_greedy
greedy_data = rb_greedy(fom, reductor,
                        fom.parameters.space(0.1, 1).sample_randomly(1000),  # training set
                        rtol=1e-2)
rom = greedy_data['rom']

# estimate and compute state-space MOR error
mu = rom.parameters.parse([0.1, 0.9, 0.2, 0.3])
u = rom.solve(mu)
print(f'Error estimate: {rom.estimate_error(mu)}')
print(f'Actual error: {(fom.solve(mu) - reductor.reconstruct(u)).norm(fom.h1_0_semi_product)}')
```

### POD-Greedy method for parabolic models

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import heat_equation_example
fom = heat_equation_example()
parameter_space = fom.parameters.space(1, 100)

from pymor.parameters.functionals import ExpressionParameterFunctional
coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
from pymor.reductors.parabolic import ParabolicRBReductor
reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)

from pymor.algorithms.greedy import rb_greedy
training_set = parameter_space.sample_uniformly(20)
greedy_data = rb_greedy(fom, reductor, training_set=parameter_space.sample_uniformly(20), max_extensions=10)
rom = greedy_data['rom']
```

### Estimation of coercivity and continuity constants using the min/max-theta approach

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example(diameter=1/10)

from pymor.parameters.functionals import ExpressionParameterFunctional
mu = fom.parameters.parse([0.1, 0.9, 0.2, 0.3])
exact_coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)

from pymor.parameters.functionals import MaxThetaParameterFunctional, MinThetaParameterFunctional
mu_bar = fom.parameters.parse([0.5, 0.5, 0.5, 0.5])
coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar, alpha_mu_bar=0.5)
continuity_estimator = MaxThetaParameterFunctional(fom.operator.coefficients, mu_bar, gamma_mu_bar=0.5)

print(f"Exact coercivity constant estimate: {exact_coercivity_estimator.evaluate(mu)}")
print(f"Coercivity constant estimate using min-theta approach: {coercivity_estimator.evaluate(mu)}")
print(f"Continuity constant estimate using max-theta approach: {continuity_estimator.evaluate(mu)}")
```

### Estimation of coercivity constants using the successive constraints method

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example(diameter=0.1)
parameter_space = fom.parameters.space(0.1, 1.)

from pymor.algorithms.scm import construct_scm_functionals
initial_parameter = parameter_space.sample_randomly(1)[0]
training_set = parameter_space.sample_randomly(50)
coercivity_estimator, _, _ = construct_scm_functionals(
    fom.operator, training_set, initial_parameter, product=fom.h1_0_semi_product, max_extensions=10, M=5)
```

### POD/neural network approximation

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example(diameter=1/10)

# instantiate reductor with training and validation parameters and desired errors
from pymor.reductors.neural_network import NeuralNetworkReductor
reductor = NeuralNetworkReductor(fom,
                                 training_set=fom.parameters.space(0.1, 1).sample_uniformly(2),
                                 validation_set=fom.parameters.space(0.1, 1).sample_randomly(5),
                                 ann_mse=None, scale_outputs=True)
rom = reductor.reduce(restarts=5)
```

### Empirical interpolation of coefficient functions

```{code-cell} ipython3
:tags: [remove-output]
from matplotlib import pyplot as plt
import numpy as np
from pymor.algorithms.ei import interpolate_function
from pymor.analyticalproblems.functions import ExpressionFunction

f = ExpressionFunction('1 + x[0]**exp[0]', 1, {'exp': 1})
parameter_space = f.parameters.space(1, 3)
f_ei, _ = interpolate_function(
    f, parameter_space.sample_uniformly(10), np.linspace(0, 1, 100).reshape((-1,1)), rtol=1e-4)

mu = f.parameters.parse(2.3)
X = np.linspace(0, 1, 100)
plt.plot(X, f(X.reshape((-1,1)), mu=mu) - f_ei(X.reshape((-1,1)), mu=mu))
plt.show()
```

### EI-Greedy/POD-Greedy reduction of nonlinear models

```{code-cell} ipython3
:tags: [remove-output]
from pymor.algorithms.ei import interpolate_operators
from pymor.algorithms.greedy import rb_greedy
from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin.fv import discretize_instationary_fv
from pymor.reductors.basic import InstationaryRBReductor

problem = burgers_problem_2d()
fom, _ = discretize_instationary_fv(problem, diameter=1./20, num_flux='engquist_osher', nt=100)
fom.enable_caching('disk')  # cache solution snapshots on disk

training_set = problem.parameter_space.sample_uniformly(10)
fom_ei, _ = interpolate_operators(
    fom, ['operator'], training_set, error_norm=fom.l2_norm, max_interpolation_dofs=30)
reductor = InstationaryRBReductor(fom_ei)
greedy_data = rb_greedy(
    fom, reductor, training_set, use_error_estimator=False, max_extensions=10)
rom = greedy_data['rom']
```

## LTI System MOR

### Available models

```{admonition} {{LTIModel}}
:class: dropdown

**Description**
This class describes input-state-output systems given by:

$$
E(\mu) \dot{x}(t, \mu) = A(\mu) x(t, \mu) + B(\mu) u(t), \\
y(t, \mu) = C(\mu) x(t, \mu) + D(\mu) u(t)
$$

if continuous-time, or

$$
E(\mu) x(k + 1, \mu) = A(\mu) x(k, \mu) + B(\mu) u(k), \\
y(k, \mu) = C(\mu) x(k, \mu) + D(\mu) u(k)
$$

if discrete-time, where $A$, $B$, $C$, $D$, and $E$ are linear operators.

**Initializing the LTIModel Class**

```python
import numpy as np
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyMatrixOperator

# Define mandatory operators A, B, and C
A = NumpyMatrixOperator(np.array([[1, 2], [3, 4]]))  # The Operator A
B = NumpyMatrixOperator(np.array([[1], [0]]))        # The Operator B
C = NumpyMatrixOperator(np.array([[0, 1]]))          # The Operator C

# Other parameters
D = NumpyMatrixOperator(np.array([[0]]))             # The Operator D
E = NumpyMatrixOperator(np.array([[1, 0], [0, 1]]))  # The Operator E
sampling_time = 0.01  # Discrete-time system with sampling time of 0.01s
T = 10                # Final time
initial_data = A.source.zeros(1) # Initial 1D data

# Initialize with only mandatory parameters
fom_basic = LTIModel(A=A, B=B, C=C)

# Initialize with initial data and time configuration
fom_discrete = LTIModel(A=A, B=B, C=C, D=D, E=E, sampling_time=sampling_time, T=T, initial_data=initial_data)

print('LTI Model with only mandatory parameters: \n{}\n'.format(fom_basic))
print('LTI Model with input data and time configuration: \n{}\n'.format(fom_discrete))
```

```{admonition} {{PHLTIModel}}
:class: dropdown

**Description**
This class describes input-state-output systems given by:

$$
E(\mu) \dot{x}(t, \mu) = (J(\mu) - R(\mu)) Q(\mu)   x(t, \mu) + (G(\mu) - P(\mu)) u(t), \\
y(t, \mu) = (G(\mu) + P(\mu))^T Q(\mu) x(t, \mu) + (S(\mu) - N(\mu)) u(t),
$$

where $H(\mu) = Q(\mu)^T E(\mu)$

$$
\Gamma(\mu) =
        \begin{bmatrix}
            J(\mu) & G(\mu) \\
            -G(\mu)^T & N(\mu)
        \end{bmatrix},
        \text{ and }
        \mathcal{W}(\mu) =
        \begin{bmatrix}
            R(\mu) & P(\mu) \\
            P(\mu)^T & S(\mu)
        \end{bmatrix}
$$

satisfy $H(\mu) = H(\mu)^T\succ 0$, $\Gamma(\mu)^T = -\Gamma(\mu)$, $\mathcal{W}(\mu) = \mathcal{W}(\mu)^T\succcurlyeq 0$

**Initializing the PHLTIModel Class**

```python
from pymor.models.iosys import PHLTIModel
from pymor.operators.numpy import NumpyMatrixOperator
import numpy as np

# Define mandatory operators
J = NumpyMatrixOperator(np.array([[0, 1], [-1, 0]]))  # Parameter J
R = NumpyMatrixOperator(np.array([[1, 0], [0, 1]]))   # Parameter R
G = NumpyMatrixOperator(np.array([[1, 1], [1, 1]]))   # Parameter G

# Define optional parameters
P = NumpyMatrixOperator(np.array([[0, 0], [0, 0]]))   # Parameter P
S = NumpyMatrixOperator(np.array([[1, 0], [1, 0]]))   # Parameter S
N = NumpyMatrixOperator(np.array([[0, 1], [2, 0]]))   # Parameter N
E = NumpyMatrixOperator(np.array([[1, 0], [0, 1]]))   # Parameter E
Q = NumpyMatrixOperator(np.array([[1, 0], [0, 1]]))   # Parameter Q
solver_options = {'lyap_lrcf': 'scipy'}               # Option to solve Lyapunov equations

# Initialize with only mandatory parameters
fom_basic = PHLTIModel(J=J, R=R, G=G)

# Initialize with addtional parameters
fom_detailed = PHLTIModel(J=J, R=R, G=G, P=P, S=S, N=N, E=E, Q=Q, solver_options=solver_options)

print('PHLTI Model with only mandatory parameters: \n{}\n'.format(fom_basic))
print('PHLTI Model with additional parameters: \n{}\n'.format(fom_detailed))
```

```{admonition} BilinearModel
:class: dropdown

**Description**
Class for bilinear systems. This class describes input-output systems given by:

$$
E x'(t) & = A x(t) + \sum_{i = 1}^m{N_i x(t) u_i(t)} + B u(t), \\
y(t) & = C x(t) + D u(t),
$$

if continuous-time, or

$$
E x(k + 1) & = A x(k) + \sum_{i = 1}^m{N_i x(k) u_i(k)} + B u(k), \\
y(k) & = C x(k) + D u(t),
$$

if discrete-time, where $E$, $A$, $N_i$, $B$, $C$, and $D$ are linear operators and $m$ is the number of inputs.

**Initializing the BilinearModel Class**

```python
from pymor.models.iosys import BilinearModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import IdentityOperator, ZeroOperator

# Define the vector, input and output spaces
vector_space = NumpyVectorSpace(3)  # Example space with dimension 3
input_space = NumpyVectorSpace(2)  # Input space with dimension 2
output_space = NumpyVectorSpace(1)  # Output space with dimension 1

# Mandatory parameters
A = IdentityOperator(vector_space)  # Operator A
N = (IdentityOperator(vector_space), IdentityOperator(vector_space))  # Tuple of N_i operators
B = ZeroOperator(vector_space, input_space)  # Operator B
C = ZeroOperator(output_space, vector_space)  # Operator C
D = ZeroOperator(output_space, input_space)  # Operator D

# Optional parameters
E = IdentityOperator(vector_space)  # Operator E (identity operator for vector space)
sampling_time = 0.1  # Positive number for discrete-time system (sampling time in seconds)

# Initialize a continuous model
fom_continuous = BilinearModel(A=A, N=N, B=B, C=C, D=D)

# Initialize a discrete model
fom_discrete = BilinearModel(A=A, N=N, B=B, C=C, D=D, E=E, sampling_time=sampling_time)

print('Continouous BilinearModel: \n{}\n'.format(fom_continuous))
print('Discrete BilinearModel: \n{}\n'.format(fom_discrete))
```

```{admonition} {{LinearDelayModel}}
:class: dropdown

**Description**
Class for linear delay systems. This class describes input-state-output systems given by:

$$
E x'(t) & = A x(t) + \sum_{i = 1}^q{A_i x(t - \tau_i)} + B u(t), \\
y(t) & = C x(t) + D u(t),
$$

if continuous-time, or

$$
E x(k + 1) & = A x(k) + \sum_{i = 1}^q{A_i x(k - \tau_i)} + B u(k), \\
y(k) & = C x(k) + D u(k),
$$

if discrete-time, where $E$, $A$, $A_i$, $B$, $C$, and $D$ are linear operators.

**Initializing the LinearDelayModel Class**

```python
from pymor.models.iosys import LinearDelayModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import IdentityOperator, ZeroOperator

# Define the vector, input and output spaces
vector_space = NumpyVectorSpace(3)  # Example state space with dimension 3
input_space = NumpyVectorSpace(3)  # Input space with dimension 3
output_space = NumpyVectorSpace(1)  # Output space with dimension 1

# Mandatory parameters
A = IdentityOperator(vector_space)  # Operator A
Ad = (IdentityOperator(vector_space), IdentityOperator(vector_space))  # Tuple of delay operators (length must match tau)
tau = (1.0, 2.0)  # Delay times (must match the length of Ad)
B = ZeroOperator(input_space, vector_space)  # Operator B
C = ZeroOperator(output_space, vector_space)  # Operator C

# Additional parameters
D = ZeroOperator(output_space, input_space)  # Operator D
E = IdentityOperator(vector_space)  # Operator E
sampling_time = 0.1  # Sampling time

# Initialize a continuous model
fom_continuous = LinearDelayModel(A=A, Ad=Ad, tau=tau, B=B, C=C)

# Initialize a discrete model
fom_discrete = LinearDelayModel(A=A, Ad=Ad, tau=tau, B=B, C=C, D=D, E=E, sampling_time=sampling_time)

print('Continouous LinearDelayModel: \n{}\n'.format(fom_continuous))
print('Discrete LinearDelayModel: \n{}\n'.format(fom_discrete))
```

```{admonition} LinearStochasticModel
:class: dropdown

**Description**
This class describes input-state-output systems given by:

$$
E \mathrm{d}x(t) & = A x(t) \mathrm{d}t + \sum_{i = 1}^q{A_i x(t) \mathrm{d}\omega_i(t)} + B u(t) \mathrm{d}t, \\
y(t) & = C x(t) + D u(t)
$$

if continuous-time, or

$$
E x(k + 1) & = A x(k) + \sum_{i = 1}^q{A_i x(k) \omega_i(k)} + B u(k), \\
y(k) & = C x(k) + D u(t)
$$

if discrete-time, where $E$, $A$, $A_i$, $B$, $C$, and $D$ are linear operators and $\omega_i$ are stochastic processes.

**Initializing the LinearStochasticModel Class**

```python
from pymor.models.iosys import LinearStochasticModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import IdentityOperator, ZeroOperator

# Define the vector, input and output spaces
vector_space = NumpyVectorSpace(3)  # Example vector space with dimension 3
input_space = NumpyVectorSpace(3)  # Input space with dimension 3
output_space = NumpyVectorSpace(1)  # Output space with dimension 1

# Continuous System
A = IdentityOperator(vector_space)  # System matrix A
As = (IdentityOperator(vector_space), IdentityOperator(vector_space))  # Tuple of stochastic operators
B = ZeroOperator(input_space, vector_space)  # Operator B
C = ZeroOperator(output_space, vector_space)  # Operator C

# Discrete System
D = ZeroOperator(output_space, input_space)  # Operator D
E = IdentityOperator(vector_space)  # Operator E
sampling_time = 1.0  # Discrete-time system with sampling time of 1 second

# Initialize a continuous model
fom_continuous = LinearStochasticModel(A=A, As=As, B=B, C=C)

# Initialize a discrete model
fom_discrete = LinearStochasticModel(A=A, As=As, B=B, C=C, D=D, E=E, sampling_time=sampling_time)

print('Continuous LinearStochasticModel: \n{}\n'.format(fom_continuous))
print('Discrete LinearStochasticModel: \n{}\n'.format(fom_discrete))
```

```{admonition} {{SecondOrderModel}}
:class: dropdown

**Description**
This class describes input-output systems given by

$$
M(\mu) \ddot{x}(t, \mu) + E(\mu) \dot{x}(t, \mu) + K(\mu) x(t, \mu)
        & = B(\mu) u(t), \\
y(t, \mu) & = C_p(\mu) x(t, \mu) + C_v(\mu) \dot{x}(t, \mu) + D(\mu) u(t),
$$

if continuous-time, or

$$
M(\mu) x(k + 2, \mu) + E(\mu) x(k + 1, \mu) + K(\mu) x(k, \mu)
        & = B(\mu) u(k), \\
y(k, \mu) & = C_p(\mu) x(k, \mu) + C_v(\mu) x(k + 1, \mu) + D(\mu) u(k)
$$

if discrete-time, where $M$, $E$, $K$, $B$, $C_p$, $C_v$, $D$ are linear operators.

**Initializing the SecondOrderModel Class**

```python
from pymor.models.iosys import SecondOrderModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator, ZeroOperator, VectorOperator
import numpy as np

vector_space = NumpyVectorSpace(3)
input_space = NumpyVectorSpace(2)
output_space = NumpyVectorSpace(1)

# Continuous System
M = IdentityOperator(vector_space)  # Mass matrix (3x3)
E = IdentityOperator(vector_space)  # Damping matrix (3x3)
K = IdentityOperator(vector_space)  # Stiffness matrix (3x3)
B = VectorOperator(vector_space.from_numpy([1, 0, 1])) # B
Cp = NumpyMatrixOperator(np.array([1, 0, 0])) # Cp

# Discrete System
Cv = NumpyMatrixOperator(np.array([1, 1, 1])) # Cv
D = ZeroOperator(output_space, B.source) # D-matrix
sampling_time = 0.01  # introduce time discretization

# Initialize a continuous model
fom_continuous = SecondOrderModel(M=M, E=E, K=K, B=B, Cp=Cp)

# Initialize a discrete model
fom_discrete = SecondOrderModel(M=M, E=E, K=K, B=B, Cp=Cp, Cv=Cv, D=D, sampling_time=sampling_time)

print('Continuous SecondOrderModel: \n{}\n'.format(fom_continuous))
print('Discrete SecondOrderModel: \n{}\n'.format(fom_discrete))
```

```{admonition} {{TransferFunction}}
:class: dropdown

**Description**
This class describes input-output systems given by a (parametrized) transfer function: $H(s,\mu)$

**Initializing the TransferFunction Class**

```python
from pymor.models.transfer_function import TransferFunction
import numpy as np

# Define the transfer function as a callable
def tf(s, mu=None):
    # Example transfer function H(s)
    return np.array([[1 / (s + 1), 0], [0, 1 / (s + 2)]])

dim_input = 2 # input dimension
dim_output = 1 # output dimension

parameters = {'mu': 2}
sampling_time = 5 # Sampling time for a discrete-time system

# Initialize continuous TransferFunction
fom_continuous = TransferFunction(dim_input=dim_input, dim_output=dim_output, tf=tf)

# Initialize discrete TransferFunction
fom_discrete = TransferFunction(dim_input=dim_input, dim_output=dim_output, tf=tf, parameters=parameters, sampling_time=sampling_time)

print('Continuous TransferFunction: \n{}\n'.format(fom_continuous))
print('Discrete TransferFunction: \n{}\n'.format(fom_discrete))
```

```{admonition} FactorizedTransferFunction
:class: dropdown

**Description**
This class describes input-output systems given by a transfer function of the form:
$H(s, \mu) = \mathcal{C}(s, \mu) \mathcal{K}(s, \mu)^{-1} \mathcal{B}(s, \mu) + \mathcal{D}(s, \mu)$

**Initializing the FactorizedTransferFunction Class**

```python
from pymor.models.transfer_function import FactorizedTransferFunction
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import IdentityOperator, VectorOperator
import numpy as np

# Dimensions
dim_input = 2
dim_output = 1
vector_space = NumpyVectorSpace(2)

# Functions that return Operators
def K(s):
    r"""Example function returning an Operator K(s)"""
    return IdentityOperator(vector_space) * (s + 1)

def B(s):
    r"""Example function returning an Operator B(s)"""
    return VectorOperator(vector_space.from_numpy(np.array([[1], [0]])))

def C(s):
    r"""Example function returning an Operator C(s)"""
    return VectorOperator(vector_space.from_numpy(np.array([[0, 1]])))

def D(s):
    r"""Example function returning an Operator D(s)"""
    return VectorOperator(vector_space.zeros(1))

# Initialize continuous FactorizedTransferFunction
fom_continuous = FactorizedTransferFunction(dim_input=dim_input, dim_output=dim_output, K=K, B=B, C=C, D=D)

# Initialize discrete FactorizedTransferFunction
fom_discrete = FactorizedTransferFunction(dim_input=dim_input, dim_output=dim_output, K=K, B=B, C=C, D=D, sampling_time=5)

print('Continuous FactorizedTransferFunction: \n{}\n'.format(fom_continuous))
print('Discrete FactorizedTransferFunction: \n{}\n'.format(fom_discrete))
```

Here we consider some of the methods for {{LTIModels}}.

### Balancing-based MOR

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import penzl_example
fom = penzl_example()

from pymor.reductors.bt import BTReductor
rom_bt = BTReductor(fom).reduce(10)

from pymor.reductors.bt import LQGBTReductor
rom_lqgbt = LQGBTReductor(fom).reduce(10)
```

### Interpolation-based MOR

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import penzl_example
fom = penzl_example()

from pymor.reductors.h2 import IRKAReductor
rom_irka = IRKAReductor(fom).reduce(10)

from pymor.reductors.h2 import TFIRKAReductor
rom_irka = TFIRKAReductor(fom).reduce(10)
```

### Eigenvalue-based MOR

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import penzl_example
fom = penzl_example()

from pymor.reductors.mt import MTReductor
rom_mt = MTReductor(fom).reduce(10)
```

### Data-driven MOR

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import penzl_example
import numpy as np
fom = penzl_example()
s = np.logspace(1, 3, 100) * 1j

from pymor.reductors.aaa import PAAAReductor
rom_aaa = PAAAReductor(s, fom).reduce()

from pymor.reductors.loewner import LoewnerReductor
rom_loewner = LoewnerReductor(s, fom).reduce()
```

Download the code:
{download}`mor_methods.md`,
{nb-download}`mor_methods.ipynb`.
