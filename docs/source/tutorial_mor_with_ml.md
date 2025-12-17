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

```{try_on_binder}
```

```{code-cell} ipython3
:load: myst_code_init.py
:tags: [remove-cell]


```

# Tutorial: Model order reduction with machine learning methods

Recent success of machine learning methods such as artificial neural networks or
kernel approaches led to the development of several
methods for model order reduction using machine learning surrogates. pyMOR provides the
functionality for a simple approach developed by Hesthaven and Ubbiali in {cite}`HU18`.
For training and evaluation of the neural networks, [PyTorch](<https://pytorch.org>) is used.
Kernel methods are implemented in pyMOR based on the vectorial kernel orthogonal
greedy algorithm (VKOGA), see {class}`~pymor.algorithms.ml.vkoga.regressor.VKOGARegressor`.

In this tutorial we will learn about feedforward neural networks and kernel greedy methods,
the basic idea of the approach by Hesthaven and Ubbiali, and how to use it in pyMOR.

## Feedforward neural networks

We aim at approximating a mapping {math}`h\colon\mathcal{P}\rightarrow Y`
between some input space {math}`\mathcal{P}\subset\mathbb{R}^p` (in our case the
parameter space) and an output space {math}`Y\subset\mathbb{R}^m` (in our case the
reduced space), given a set {math}`S=\{(\mu_i,h(\mu_i))\in\mathcal{P}\times Y: i=1,\dots,N\}`
of samples, by means of an artificial neural network. In this context, neural
networks serve as a special class of functions that are able to "learn" the
underlying structure of the sample set {math}`S` by adjusting their weights.
More precisely, feedforward neural networks consist of several layers, each
comprising a set of neurons that are connected to neurons in adjacent layers.
A so-called "weight" is assigned to each of those connections. The weights in
the neural network can be adjusted while fitting the neural network to the
given sample set. For a given input {math}`\mu\in\mathcal{P}`, the weights between the
input layer and the first hidden layer (the one after the input layer) are
multiplied with the respective values in {math}`\mu` and summed up. Subsequently,
a so-called "bias" (also adjustable during training) is added and the result is
assigned to the corresponding neuron in the first hidden layer. Before passing
those values to the following layer, a (non-linear) activation function
{math}`\rho\colon\mathbb{R}\rightarrow\mathbb{R}` is applied. If {math}`\rho`
is linear, the function implemented by the neural network is affine, since
solely affine operations were performed. Hence, one usually chooses a
non-linear activation function to introduce non-linearity in the neural network
and thus increase its approximation capability. In some sense, the input
{math}`\mu` is passed through the neural network, affine-linearly combined with the
other inputs and non-linearly transformed. These steps are repeated in several
layers.

The following figure shows a simple example of a neural network with two hidden
layers, an input size of two and an output size of three. Each edge between
neurons has a corresponding weight that is learnable in the training phase.

```{image} neural_network.svg
:alt: Feedforward neural network
:width: 100%
```

To train the neural network, one considers a so-called "loss function", that
measures how the neural network performs on the training parameters {math}`S`, i.e.
how accurately the neural network reproduces the output {math}`h(\mu_i)` given
the input {math}`\mu_i`. The weights of the neural network are adjusted
iteratively such that the loss function is successively minimized. To this end,
one typically uses a Quasi-Newton method for small neural networks or a
(stochastic) gradient descent method for deep neural networks (those with many
hidden layers).

In pyMOR, there exists a training routine for neural networks. This
procedure is part of the `fit`-method of the
{class}`~pymor.algorithms.ml.nn.regressor.NeuralNetworkRegressor`
and it is not necessary to write a custom training algorithm for each specific
problem. The training data is automatically split in a random fashion into
training and validation set. However, it is sometimes necessary to try
different architectures for the neural network to find the one that best fits
the problem at hand. In the regressor, one can easily adjust the number of
layers and the number of neurons in each hidden layer, for instance.
Furthermore, it is also possible to change the deployed activation function.

A possibility to use feedforward neural networks in combination with reduced
basis methods will be discussed below. First, we introduce a different machine
learning technique based on kernel interpolation. Both methods can be used within
pyMOR for model order reduction. It is further possible to employ any regressor
for regression problems from scikit-learn.

## Greedy kernel methods

## A non-intrusive reduced order method using machine learning

We now assume that we are given a parametric pyMOR {{ Model }} for which we want
to compute a reduced order surrogate {{ Model }} using a machine learning method.
In this example, we consider the following two-dimensional diffusion problem with
parametrized diffusion, right hand side and Dirichlet boundary condition:

```{math}
-\nabla \cdot \big(\sigma(x, \mu) \nabla u(x, \mu) \big) = f(x, \mu),\quad x=(x_1,x_2) \in \Omega,
```

on the domain {math}`\Omega:= (0, 1)^2 \subset \mathbb{R}^2` with data
functions {math}`f((x_1, x_2), \mu) = 10 \cdot \mu + 0.1`,
{math}`\sigma((x_1, x_2), \mu) = (1 - x_1) \cdot \mu + x_1`, where
{math}`\mu \in (0.1, 1)` denotes the parameter. Further, we apply the
Dirichlet boundary conditions

```{math}
u((x_1, x_2), \mu) = 2x_1\mu + 0.5,\quad x=(x_1, x_2) \in \partial\Omega.
```

We discretize the problem using pyMOR's built-in discretization toolkit as
explained in {doc}`tutorial_builtin_discretizer`:

```{code-cell} ipython3
from pymor.basic import *

problem = StationaryProblem(
      domain=RectDomain(),

      rhs=LincombFunction(
          [ExpressionFunction('10', 2), ConstantFunction(1., 2)],
          [ProjectionParameterFunctional('mu'), 0.1]),

      diffusion=LincombFunction(
          [ExpressionFunction('1 - x[0]', 2), ExpressionFunction('x[0]', 2)],
          [ProjectionParameterFunctional('mu'), 1]),

      dirichlet_data=LincombFunction(
          [ExpressionFunction('2 * x[0]', 2), ConstantFunction(1., 2)],
          [ProjectionParameterFunctional('mu'), 0.5]),

      name='2DProblem'
  )

fom, _ = discretize_stationary_cg(problem, diameter=1/50)
```

Since we employ a single {{ Parameter }}, and thus use the same range for each
parameter, we can create the {{ ParameterSpace }} using the following line:

```{code-cell} ipython3
parameter_space = fom.parameters.space((0.1, 1))
```

The main idea of the approach by Hesthaven et al. is to approximate the mapping
from the {{ Parameters }} to the coefficients of the respective solution in a
reduced basis by means of a neural network. Thus, in the online phase, one
performs a forward pass of the {{ Parameters }} through the neural networks and
obtains the approximated reduced coordinates. To derive the corresponding
high-fidelity solution, one can further use the reduced basis and compute the
linear combination defined by the reduced coefficients. The reduced basis is
created via POD.

The method described above is "non-intrusive", which means that no deep insight
into the model or its implementation is required and it is completely
sufficient to be able to generate full order snapshots for a randomly chosen
set of parameters. This is one of the main advantages of the proposed approach,
since one can simply train a neural network, check its performance and resort
to a different method if the neural network does not provide proper
approximation results.

Further, the method is actually independent of the particular machine learning
approach. It is therefore possible to use, for instance, kernel methods instead
of neural networks as originally proposed in {cite}`HU18`. In pyMOR, the implementation
can deal with any regressor fulfilling the scikit-learn interface. The neural networks
and the kernel methods are implemented in pyMOR in such a way that they also follow
the scikit-learn interface and the reductor only requires such an regressor.
In this tutorial, we will compare neural networks and kernel methods and show
how they can be trained and used in the context of model order reduction.

To train the machine learning surrogates, we create a set of training parameters
consisting of 100 randomly chosen {{ parameter_values }}:

```{code-cell} ipython3
training_parameters = parameter_space.sample_uniformly(100)
```

In this tutorial, we construct the reduced basis such that no more modes than
required to bound the l2-approximation error by a given value are used.
The l2-approximation error is the error of the orthogonal projection (in the
l2-sense) of the training snapshots onto the reduced basis. That is, we
prescribe `l2_err` in the POD method. It is also possible to determine a relative
or absolute tolerance (in the singular values) that should not be exceeded on
the training parameters. Further, one can preset the size of the reduced basis.
The construction of the reduced basis is independent of the machine learning
surrogate and is therefore not part of the reductor. The reduced basis has to be
computed beforehand and provided (together with the reduced coeffcients, for
instance the coefficients with respect to the reduced basis of the orthogonal
projection onto the reduced space) to the reductor. Within the reductor, mainly
the training of the regressor using the correct data formats is performed and
suitable reduced models are constructed.

We start by collecting the training snapshots associated to the training parameters:

```{code-cell} ipython3
training_snapshots = fom.solution_space.empty(reserve=len(training_parameters))
for mu in training_parameters:
    training_snapshots.append(fom.solve(mu))
```

Afterwards, we compute a reduced basis using POD

```{code-cell} ipython3
RB, _ = pod(training_snapshots, l2_err=1e-5)
```

and project the training snapshots onto the reduced basis to obtain the
training data for the machine learning surrogates:

```{code-cell} ipython3
projected_training_snapshots = training_snapshots.inner(RB)
```

We now initialize regressors for feedforward neural networks

```{code-cell} ipython3
from pymor.algorithms.ml.nn import FullyConnectedNN, NeuralNetworkRegressor
neural_network = FullyConnectedNN(hidden_layers=[30, 30, 30])
nn_regressor = NeuralNetworkRegressor(neural_network, tol=1e-4)
```

and kernel methods

```{code-cell} ipython3
from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGARegressor
kernel = GaussianKernel(length_scale=1.0)
vkoga_regressor = VKOGARegressor(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
```

Finally, we construct data-driven reductors using the different regressors
and call the respective `reduce`-method to start the training process:

```{code-cell} ipython3
from pymor.reductors.data_driven import DataDrivenReductor
nn_reductor = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                 regressor=nn_regressor, reduced_basis=RB)
nn_rom = nn_reductor.reduce()

vkoga_reductor = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                    regressor=vkoga_regressor, reduced_basis=RB)
vkoga_rom = vkoga_reductor.reduce()
```

We are now ready to test our reduced models by solving for a random parameter value
the full problem and the reduced models and visualize the result:

```{code-cell} ipython3
mu = parameter_space.sample_randomly()

U = fom.solve(mu)
# Neural network based model
U_red_nn = nn_rom.solve(mu)
U_red_nn_recon = nn_reductor.reconstruct(U_red_nn)
# Kernel based model
U_red_vkoga = vkoga_rom.solve(mu)
U_red_vkoga_recon = vkoga_reductor.reconstruct(U_red_vkoga)

fom.visualize((U, U_red_nn_recon, U_red_vkoga_recon),
              legend=(f'Full solution for parameter {mu}', f'Reduced solution using NN for parameter {mu}',
                      f'Reduced solution using VKOGA for parameter {mu}'))
```

Finally, we measure the error of our neural network and kernel surrogates
and the performance in terms of computational speedup compared to the
solution of the full order problem for some test parameters.
To this end, we sample randomly
some {{ parameter_values }} from our {{ ParameterSpace }}:

```{code-cell} ipython3
test_parameters = parameter_space.sample_randomly(10)
```

Next, we create empty solution arrays for the full and reduced solutions and an
empty list for the speedups:

```{code-cell} ipython3
U = fom.solution_space.empty(reserve=len(test_parameters))
U_red_nn = fom.solution_space.empty(reserve=len(test_parameters))
U_red_vkoga = fom.solution_space.empty(reserve=len(test_parameters))

speedups_nn = []
speedups_vkoga = []
```

Now, we iterate over the test parameters, compute full and reduced solutions to the
respective parameters and measure the speedup:

```{code-cell} ipython3
import time

for mu in test_parameters:
    tic = time.perf_counter()
    U.append(fom.solve(mu))
    time_fom = time.perf_counter() - tic

    # Neural network based model
    tic = time.perf_counter()
    U_red_nn.append(nn_reductor.reconstruct(nn_rom.solve(mu)))
    time_red_nn = time.perf_counter() - tic
    speedups_nn.append(time_fom / time_red_nn)

    # Kernel based model
    tic = time.perf_counter()
    U_red_vkoga.append(vkoga_reductor.reconstruct(vkoga_rom.solve(mu)))
    time_red_vkoga = time.perf_counter() - tic
    speedups_vkoga.append(time_fom / time_red_vkoga)
```

We can now derive the absolute and relative errors on the training parameters as

```{code-cell} ipython3
absolute_errors_nn = (U - U_red_nn).norm()
relative_errors_nn = absolute_errors_nn / U.norm()

absolute_errors_vkoga = (U - U_red_vkoga).norm()
relative_errors_vkoga = absolute_errors_vkoga / U.norm()
```

The average absolute errors amount to

```{code-cell} ipython3
import numpy as np

print(f"Neural network: {np.average(absolute_errors_nn)}")
print(f"Kernel method: {np.average(absolute_errors_vkoga)}")
```

On the other hand, the average relative errors are

```{code-cell} ipython3
print(f"Neural network: {np.average(relative_errors_nn)}")
print(f"Kernel method: {np.average(relative_errors_vkoga)}")
```

Using machine learning results in the following median speedups compared to
solving the full order problem:

```{code-cell} ipython3
print(f"Neural network: {np.median(speedups_nn)}")
print(f"Kernel: {np.median(speedups_vkoga)}")
```

Since {class}`~pymor.reductors.data_driven.DataDrivenReductor` only uses the provided
training data, the approach presented here can easily be applied to {{ Models }}
originating from external solvers, without requiring any access to {{ Operators }}
internal to the solver. Examples using FEniCS for stationary and instationary problems
together with the {class}`~pymor.reductors.data_driven.DataDrivenReductor` are provided
in {mod}`~pymordemos.data_driven_fenics` and {mod}`~pymordemos.data_driven_instationary`.
Furthermore, the stratedy is also applicable when no full-order model is available at all.
Given a set of training snapshots (for instance read from a file), a reduced basis can be
computed using a data-driven compression method such as POD, the snapshots can be
projected onto the reduced basis and the machine learning training is handled by the
data-driven reductor as shown before.

## Direct approximation of output quantities

Thus far, we were mainly interested in approximating the solution state
{math}`u(\mu)\equiv u(\cdot,\mu)` for some parameter {math}`\mu`. If we consider an output
functional {math}`\mathcal{J}(\mu):= J(u(\mu), \mu)`, one can use the reduced solution
{math}`u_N(\mu)` for computing the output as {math}`\mathcal{J}(\mu)\approx J(u_N(\mu),\mu)`.
However, when dealing with neural networks, one could also think about directly learning the
mapping from parameter to output. That is, one can use a neural network to approximate
{math}`\mathcal{J}\colon\mathcal{P}\to\mathbb{R}^q`, where {math}`q\in\mathbb{N}` denotes
the output dimension.

In the following, we will extend our problem from the last section by an output functional
and use the {class}`~pymor.reductors.data_driven.DataDrivenReductor` with the argument
`target_quantity='output'` to derive a reduced model that can solely be used to solve
for the output quantity without computing a reduced state at all.

For the definition of the output, we define the output of out problem as the l2-product of the
solution with the right hand side respectively Dirichlet boundary data of our original problem:

```{code-cell} ipython3
problem = problem.with_(outputs=[('l2', problem.rhs), ('l2_boundary', problem.dirichlet_data)])
```

Consequently, the output dimension is {math}`q=2`. After adjusting the problem definition,
we also have to update the full order model to be aware of the output quantities:

```{code-cell} ipython3
fom, _ = discretize_stationary_cg(problem, diameter=1/50)
```

We can now use again the {class}`~pymor.reductors.data_driven.DataDrivenReductor`
(for simplicity we only consider kernel methods here) and initialize the reductor
using output data:

```{code-cell} ipython3
training_outputs = []
for mu in training_parameters:
    training_outputs.append(fom.output(mu)[:, 0])
training_outputs = np.array(training_outputs)

vkoga_output_regressor = VKOGARegressor(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
output_reductor = DataDrivenReductor(training_parameters, training_outputs,
                                     regressor=vkoga_output_regressor, target_quantity='output')
```

Observe that we now specified `target_quantity='output'` instead of the default value
`target_quantity='solution'` when creating the reductor. On the other hand, we do not need
a reduced basis now since we are solely interested in an approximation of the output.

Similar to the {class}`~pymor.reductors.data_driven.DataDrivenReductor`
with `target_quantity='solution'`, we can call `reduce` to obtain a reduced order model.
In this case, `reduce` trains the machine learning surrogate to approximate the mapping from
parameter to output directly. Therefore, we can only use the resulting reductor to solve for
the outputs and not for state approximations.
The {class}`~pymor.reductors.data_driven.DataDrivenReductor` with
`target_quantity='solution'` though can be used to do both by calling `solve`
respectively `output` (if we had initialized
the {class}`~pymor.reductors.data_driven.DataDrivenReductor` with
`target_quantity='solution'` and the problem including the output quantities).

We now perform the reduction and run some tests with the resulting
{class}`~pymor.models.data_driven.DataDrivenModel`:

```{code-cell} ipython3
output_rom = output_reductor.reduce()

outputs = []
outputs_red = []
outputs_speedups = []

for mu in test_parameters:
    tic = time.perf_counter()
    outputs.append(fom.output(mu=mu))
    time_fom = time.perf_counter() - tic

    tic = time.perf_counter()
    outputs_red.append(output_rom.output(mu=mu))
    time_red = time.perf_counter() - tic

    outputs_speedups.append(time_fom / time_red)

outputs = np.squeeze(np.array(outputs))
outputs_red = np.squeeze(np.array(outputs_red))

outputs_absolute_errors = np.abs(outputs - outputs_red)
outputs_relative_errors = outputs_absolute_errors / np.abs(outputs)
```

The average absolute error (component-wise) on the test parameters is given by

```{code-cell} ipython3
np.average(outputs_absolute_errors)
```

The average relative error is

```{code-cell} ipython3
np.average(outputs_relative_errors)
```

and the median of the speedups amounts to

```{code-cell} ipython3
np.median(outputs_speedups)
```

## Neural networks for instationary problems

To solve instationary problems using machine learning, we have extended the
{class}`~pymor.reductors.data_driven.DataDrivenReductor` to also treat instationary cases,
where time is treated either as an additional parameter (see {cite}`WHR19`) or the whole time
trajectory can be predicted at once. In the first case, the input, together
with the current time instance, is passed to the machine learning surrogate in each time step
to obtain reduced coefficients. In the second case, the parameter is used as input and the of
the machine learning surrogate is the complete time trajectory of reduced coefficients.
In the same fashion, setting `target_quantity='output'` yields a reduced model for prediction
of output trajectories without requiring information about the solution states.

A slightly different approach that is also implemented in pyMOR and uses a different type of
neural network is described in the following section.

### Long short-term memory neural networks for instationary problems

So-called *recurrent neural networks* are especially well-suited for capturing time-dependent
dynamics. These types of neural networks can treat input sequences of variable length (in our case
sequences with a variable number of time steps) and store internal states that are passed from one
time step to the next. Therefore, these networks implement an internal memory that keeps
information over time. Furthermore, for each element of the input sequence, the same neural
network is applied.

In pyMOR, a specific type of recurrent neural network is implemented, namely a so-called
*long short-term memory neural network (LSTM)*, first introduced in {cite}`HS97`, that tries to
avoid problems like vanishing or exploding gradients that often occur during training of recurrent
neural networks.

#### The architecture of an LSTM neural network

In an LSTM neural network, multiple so-called LSTM cells are chained with each other such that the
cell state {math}`c_k` and the hidden state {math}`h_k` of the {math}`k`-th LSTM cell serve as the
input hidden states for the {math}`k+1`-th LSTM cell. Therefore, information from former time
steps can be available later. Each LSTM cell takes an input {math}`\mu(t_k)` and produces an
output {math}`o(t_k)`. The following figure shows the general structure of an LSTM neural network
that is also implemented in the same way in pyMOR:

```{image} lstm.svg
:alt: Long short-term neural network
:width: 100%
```

#### The LSTM cell

The main building block of an LSTM network is the *LSTM cell*, which is denoted by {math}`\Phi`,
and sketched in the following figure:

```{image} lstm_cell.svg
:alt: LSTM cell
:align: left
```

Here, {math}`\mu(t_k)` denotes the input of the network at the current time instance {math}`t_k`,
while {math}`o(t_k)` denotes the output. The two hidden states for time instance `t_k` are given
as the cell state {math}`c_k` and the hidden state {math}`h_k` that also serves as the output.
Squares represent layers similar to those used in feedforward neural networks, where inside the
square the applied activation function is mentioned, and circles denote element-wise
operations like element-wise multiplication ({math}`\times`), element-wise addition ({math}`+`) or
element-wise application of the hyperbolic tangent function ({math}`\tanh`). The filled black
circle represents the concatenation of the inputs. Furthermore, {math}`\sigma` is the sigmoid
activation function ({math}`\sigma(x)=\frac{1}{1+\exp(-x)}`), and {math}`\tanh` is the hyperbolic
tangent activation function ({math}`\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}`) used for
the respective layers in the LSTM network. Finally, the layer {math}`P` denotes a projection layer
that projects vectors of the internal size to the hidden and output size. Hence, internally, the
LSTM can deal with larger quantities and finally projects them onto a space with a desired size.
Altogether, a single LSTM cell takes two hidden states and an input of the form
{math}`(c_{k-1},h_{k-1},\mu(t_k))` and transforms them into new hidden states and an output state
of the form {math}`(c_k,h_k,o(t_k))`.

We will take a closer look at the individual components of an LSTM cell in the subsequent
paragraphs.

##### The forget gate

```{image} lstm_cell_forget_gate.svg
:alt: Forget gate of an LSTM cell
:align: right
```

As the name already suggests, the *forget gate* determines which part of the cell state
{math}`c_{k-1}` the network forgets when moving to the next cell state {math}`c_k`. The main
component of the forget gate is a neural network layer consisting of an affine-linear function
with adjustable weights and biases followed by a sigmoid nonlinearity. By applying the sigmoid
activation function, the output of the layer is scaled to lie between 0 and 1. The cell state
{math}`c_{k-1}` from the previous cell is (point-wise) multiplied by the output of the layer in
the forget gate. Hence, small values in the output of the layer correspond to parts of the cell
state that are diminished, while values near 1 mean that the corresponding parts of the cell
state remain intact. As input of the forget gate serves the pair {math}`(h_{k-1},\mu(t_k))` and
in the second step also the cell state {math}`c_{k-1}`.

##### The input gate

```{image} lstm_cell_input_gate.svg
:alt: Input gate of an LSTM cell
:align: right
```

To further change the cell state, an LSTM cell contains a so-called *input gate*. This gate mainly
consists of two layers, a sigmoid layer and an hyperbolic tangent layer, acting on the pair
{math}`(h_{k-1},\mu(t_k))`. As in the forget gate, the sigmoid layer determines which parts of the
cell state to adjust. On the other hand, the hyperbolic tangent layer determines how to adjust the
cell state. Using the hyperbolic tangent as activation function scales the output to be between -1
and 1, and allows for small updates of the cell state. To finally compute the update, the outputs
of the sigmoid and the hyperbolic tangent layer are multiplied entry-wise. Afterwards, the update
is added to the cell state (after the cell state passed the forget gate). The new cell state is
now prepared to be passed to the subsequent LSTM cell.

##### The output gate

```{image} lstm_cell_output_gate.svg
:alt: Output gate of an LSTM cell
:align: right
```

For computing the output {math}`o(t_k)` (and the new hidden state {math}`h_k`), the updated cell
state {math}`c_k` is first of all entry-wise transformed using a hyperbolic tangent function such
that the result again takes values between -1 and 1. Simultaneously, a neural network layer with a
sigmoid activation function is applied to the concatenated pair {math}`(h_{k-1},\mu(t_k))` of
hidden state and input. Both results are multiplied entry-wise. This results in a filtered version
of the (normalized) cell state. Finally, a projection layer is applied such that the result of the
output gate has the desired size and can take arbitrary real values (before, due to the sigmoid and
hyperbolic tangent activation functions, the outcome was restricted to the interval from -1 to 1).
The projection layer applies a linear function without an activation (similar to the last layer of
a usual feedforward neural network but without bias). Altogether, the *output gate* produces an
output {math}`o(t_k)` that is returned and a new hidden state {math}`h_k` that can be passed
(together with the updated cell state {math}`c_k`) to the next LSTM cell.

#### LSTMs for model order reduction

The idea of the approach implemented in pyMOR is the following: Instead of passing the current
time instance as an additional input of the neural network, we use an LSTM that takes at each time
instance {math}`t_k` the (potentially) time-dependent input {math}`\mu(t_k)` as an input and uses
the hidden states of the former time step. The output {math}`o(t_k)` of the LSTM (and therefore
also the hidden state {math}`h_k`) at time {math}`t_k` are either approximations of the reduced
basis coefficients (if `target_quantity='solution'`) or approximations of the
output quantities (`target_quantity='output'`). In order to use LSTMs in pyMOR, one simply
initializes a {class}`~pymor.algorithms.ml.nn.neural_networks.LongShortTermMemoryNN` and
creates a {class}`~pymor.algorithms.ml.nn.regressor.NeuralNetworkRegressor` with the LSTM.
Everything else is automatically handled by pyMOR when using
the {class}`~pymor.reductors.data_driven.DataDrivenReductor`.

### Instationary neural network reductors in practice

In the following we apply different machine learning surrogates to a parametrized parabolic
equation. First, we import the parametrized heat equation example from
{mod}`~pymor.models.examples`:

```{code-cell} ipython3
from pymor.models.examples import heat_equation_example
fom = heat_equation_example()
product = fom.h1_0_semi_product
```

We further define the parameter space:

```{code-cell} ipython3
parameter_space = fom.parameters.space(1, 25)
```

Additionally, we sample training and test parameters from the respective parameter space:

```{code-cell} ipython3
training_parameters = parameter_space.sample_uniformly(15)
test_parameters = parameter_space.sample_randomly(10)
```

To check how the different reduced models perform, we write a simple function that measures
the errors and the speedups on a set of test parameters:

```{code-cell} ipython3
def compute_errors(rom, reductor):
    speedups = []

    U = fom.solution_space.empty(reserve=len(test_parameters))
    U_red = fom.solution_space.empty(reserve=len(test_parameters))

    for mu in test_parameters:
        tic = time.time()
        u_fom = fom.solve(mu)[1:]
        U.append(u_fom)
        time_fom = time.time() - tic

        tic = time.time()
        u_red = reductor.reconstruct(rom.solve(mu))[1:]
        U_red.append(u_red)
        time_red = time.time() - tic

        speedups.append(time_fom / time_red)

    relative_errors = (U - U_red).norm2() / U.norm2()

    return relative_errors, speedups
```

We now run the {class}`~pymor.reductors.data_driven.DataDrivenReductor` using
different machine learning surrogates (VKOGA, VKOGA with time-vectorization,
fully-connected neural network, LSTM) and evaluate the performance of the
resulting reduced models:

```{code-cell} ipython3
training_snapshots = fom.solution_space.empty(reserve=len(training_parameters))
for mu in training_parameters:
    training_snapshots.append(fom.solve(mu))

basis_size = 20
RB, _ = pod(training_snapshots, modes=basis_size)
projected_training_snapshots = training_snapshots.inner(RB)
```

It is often useful for the machine learning training to scale inputs and outputs,
for instance using scikit-learn's `MinMaxScaler`. This will be incorporated below
as well:

```{code-cell} ipython3
from sklearn.preprocessing import MinMaxScaler

vkoga_regressor = VKOGARegressor()
vkoga_reductor = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                    regressor=vkoga_regressor, target_quantity='solution',
                                    reduced_basis=RB, T=fom.T, time_vectorized=False,
                                    input_scaler=MinMaxScaler(), output_scaler=MinMaxScaler())
vkoga_rom = vkoga_reductor.reduce()
rel_errors_vkoga, speedups_vkoga = compute_errors(vkoga_rom, vkoga_reductor)
```

```{code-cell} ipython3
vkoga_regressor_tv = VKOGARegressor()
vkoga_reductor_tv = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                    regressor=vkoga_regressor_tv, target_quantity='solution',
                                    reduced_basis=RB, T=fom.T, time_vectorized=True,
                                    input_scaler=MinMaxScaler(), output_scaler=MinMaxScaler())
vkoga_rom_tv = vkoga_reductor_tv.reduce()
rel_errors_vkoga_tv, speedups_vkoga_tv = compute_errors(vkoga_rom_tv, vkoga_reductor_tv)
```

```{code-cell} ipython3
nn_regressor = NeuralNetworkRegressor(tol=None, restarts=0)
nn_reductor = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                 regressor=nn_regressor, target_quantity='solution',
                                 reduced_basis=RB, T=fom.T, time_vectorized=False,
                                 input_scaler=MinMaxScaler(), output_scaler=MinMaxScaler())
nn_rom = nn_reductor.reduce()
rel_errors_nn, speedups_nn = compute_errors(nn_rom, nn_reductor)
```

```{code-cell} ipython3
"""
from pymor.algorithms.ml.nn.neural_networks import LongShortTermMemoryNN
lstm_regressor = NeuralNetworkRegressor(LongShortTermMemoryNN(hidden_dimension=25, number_layers=1),
                                        tol=None, restarts=0, learning_rate=0.01)
lstm_reductor = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                   regressor=lstm_regressor, target_quantity='solution',
                                   reduced_basis=RB, T=fom.T, time_vectorized=False,
                                   input_scaler=MinMaxScaler(), output_scaler=MinMaxScaler())
lstm_rom = lstm_reductor.reduce()
rel_errors_lstm, speedups_lstm = compute_errors(lstm_rom, lstm_reductor)
"""
```

We finally print the results:

```{code-cell} ipython3
print('Results for the state approximation:')
print('====================================')
print()
print('Approach by Hesthaven and Ubbiali using feedforward ANNs:')
print('---------------------------------------------------------')
print(f'Average relative error: {np.average(rel_errors_nn)}')
print(f'Median of speedup: {np.median(speedups_nn)}')
print()
"""
print('Approach using long short-term memory ANNs:')
print('-------------------------------------------')
print(f'Average relative error: {np.average(rel_errors_lstm)}')
print(f'Median of speedup: {np.median(speedups_lstm)}')
"""
print()
print('Approach by Hesthaven and Ubbiali using VKOGA:')
print('----------------------------------------------')
print(f'Average relative error: {np.average(rel_errors_vkoga)}')
print(f'Median of speedup: {np.median(speedups_vkoga)}')
print()
print('Approach by Hesthaven and Ubbiali using VKOGA (time-vectorized):')
print('----------------------------------------------------------------')
print(f'Average relative error: {np.average(rel_errors_vkoga_tv)}')
print(f'Median of speedup: {np.median(speedups_vkoga_tv)}')
```

In this example, we observe that the LSTMs perform much better than the feedforward ANNs in terms
of accuracy while the speedups of both methods lie in the same order of magnitude.

Download the code:
{download}`tutorial_mor_with_ml.md`
{nb-download}`tutorial_mor_with_ml.ipynb`
