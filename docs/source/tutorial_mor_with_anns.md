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

# Tutorial: Model order reduction with artificial neural networks

Recent success of artificial neural networks led to the development of several
methods for model order reduction using neural networks. pyMOR provides the
functionality for a simple approach developed by Hesthaven and Ubbiali in {cite}`HU18`.
For training and evaluation of the neural networks, [PyTorch](<https://pytorch.org>) is used.

In this tutorial we will learn about feedforward neural networks, the basic
idea of the approach by Hesthaven et al., and how to use it in pyMOR.

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

A possibility to use feedforward neural networks in combination with reduced
basis methods will be introduced in the following section.

## A non-intrusive reduced order method using artificial neural networks

We now assume that we are given a parametric pyMOR {{ Model }} for which we want
to compute a reduced order surrogate {{ Model }} using a neural network. In this
example, we consider the following two-dimensional diffusion problem with
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

In pyMOR, there exists a training routine for feedforward neural networks. This
procedure is part of a reductor and it is not necessary to write a custom
training algorithm for each specific problem. However, it is sometimes
necessary to try different architectures for the neural network to find the one
that best fits the problem at hand. In the reductor, one can easily adjust the
number of layers and the number of neurons in each hidden layer, for instance.
Furthermore, it is also possible to change the deployed activation function.

To train the neural network, we create a set of training and a validation parameters
consisting of 100 and 20 randomly chosen {{ parameter_values }}, respectively:

```{code-cell} ipython3
training_parameters = parameter_space.sample_uniformly(100)
validation_parameters = parameter_space.sample_randomly(20)
```

In this tutorial, we construct the reduced basis such that no more modes than
required to bound the l2-approximation error by a given value are used.
The l2-approximation error is  the error of the orthogonal projection (in the
l2-sense) of the training snapshots onto the reduced basis. That is, we
prescribe `l2_err` in the reductor. It is also possible to determine a relative
or absolute tolerance (in the singular values) that should not be exceeded on
the training parameters. Further, one can preset the size of the reduced basis.

The training is aborted when a neural network that guarantees our prescribed
tolerance is found. If we set `ann_mse` to `None`, this function will
automatically train several neural networks with different initial weights and
select the one leading to the best results on the validation parameters. We can also
set `ann_mse` to `'like_basis'`. Then, the algorithm tries to train a neural
network that leads to a mean squared error on the training parameters that is as small
as the error of the reduced basis. If the maximal number of restarts is reached
without finding a network that fulfills the tolerances, an exception is raised.
In such a case, one could try to change the architecture of the neural network
or switch to `ann_mse=None` which is guaranteed to produce a reduced order
model (perhaps with insufficient approximation properties).

We can now construct a reductor with prescribed error for the basis and mean
squared error of the neural network:

```{code-cell} ipython3
from pymor.reductors.neural_network import NeuralNetworkReductor

reductor = NeuralNetworkReductor(fom,
                                 training_parameters=training_parameters,
                                 validation_parameters=validation_parameters,
                                 l2_err=1e-5,
                                 ann_mse=1e-5)
```

To reduce the model, i.e. compute a reduced basis via POD and train the neural
network, we use the respective function of the
{class}`~pymor.reductors.neural_network.NeuralNetworkReductor`:

```{code-cell} ipython3
rom = reductor.reduce(restarts=100)
```

We are now ready to test our reduced model by solving for a random parameter value
the full problem and the reduced model and visualize the result:

```{code-cell} ipython3
mu = parameter_space.sample_randomly()

U = fom.solve(mu)
U_red = rom.solve(mu)
U_red_recon = reductor.reconstruct(U_red)

fom.visualize((U, U_red_recon),
              legend=(f'Full solution for parameter {mu}', f'Reduced solution for parameter {mu}'))
```

Finally, we measure the error of our neural network and the performance
compared to the solution of the full order problem on the training parameters. To this
end, we sample randomly some {{ parameter_values }} from our {{ ParameterSpace }}:

```{code-cell} ipython3
test_parameters = parameter_space.sample_randomly(10)
```

Next, we create empty solution arrays for the full and reduced solutions and an
empty list for the speedups:

```{code-cell} ipython3
U = fom.solution_space.empty(reserve=len(test_parameters))
U_red = fom.solution_space.empty(reserve=len(test_parameters))

speedups = []
```

Now, we iterate over the test parameters, compute full and reduced solutions to the
respective parameters and measure the speedup:

```{code-cell} ipython3
import time

for mu in test_parameters:
    tic = time.perf_counter()
    U.append(fom.solve(mu))
    time_fom = time.perf_counter() - tic

    tic = time.perf_counter()
    U_red.append(reductor.reconstruct(rom.solve(mu)))
    time_red = time.perf_counter() - tic

    speedups.append(time_fom / time_red)
```

We can now derive the absolute and relative errors on the training parameters as

```{code-cell} ipython3
absolute_errors = (U - U_red).norm()
relative_errors = (U - U_red).norm() / U.norm()
```

The average absolute error amounts to

```{code-cell} ipython3
import numpy as np

np.average(absolute_errors)
```

On the other hand, the average relative error is

```{code-cell} ipython3
np.average(relative_errors)
```

Using neural networks results in the following median speedup compared to
solving the full order problem:

```{code-cell} ipython3
np.median(speedups)
```

Since {class}`~pymor.reductors.neural_network.NeuralNetworkReductor` only calls
the {meth}`~pymor.models.interface.Model.solve` method of the {{ Model }}, it can easily
be applied to {{ Models }} originating from external solvers, without requiring any access to
{{ Operators }} internal to the solver.

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
and use the {class}`~pymor.reductors.neural_network.NeuralNetworkStatefreeOutputReductor` to
derive a reduced model that can solely be used to solve for the output quantity without
computing a reduced state at all.

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

We can now import the {class}`~pymor.reductors.neural_network.NeuralNetworkStatefreeOutputReductor`
and initialize the reductor using the same data as before:

```{code-cell} ipython3
from pymor.reductors.neural_network import NeuralNetworkStatefreeOutputReductor

output_reductor = NeuralNetworkStatefreeOutputReductor(fom,
                                                       training_parameters=training_parameters,
                                                       validation_parameters=validation_parameters,
                                                       validation_loss=1e-5)
```

Similar to the `NeuralNetworkReductor`, we can call `reduce` to obtain a reduced order model.
In this case, `reduce` trains a neural network to approximate the mapping from parameter to
output directly. Therefore, we can only use the resulting reductor to solve for the outputs
and not for state approximations. The `NeuralNetworkReductor` though can be used to do both by
calling `solve` respectively `output` (if we had initialized the `NeuralNetworkReductor` with
the problem including the output quantities).

We now perform the reduction and run some tests with the resulting
{class}`~pymor.models.neural_network.NeuralNetworkStatefreeOutputModel`:

```{code-cell} ipython3
output_rom = output_reductor.reduce(restarts=100)

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
outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)
```

The average absolute error (component-wise) on the training parameters is given by

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

To solve instationary problems using neural networks, we have extended the
{class}`~pymor.reductors.neural_network.NeuralNetworkReductor` to also treat instationary cases, where time
is treated as an additional parameter (see {cite}`WHR19`). It passes the input, together
with the current time instance, through the neural network in each time step to obtain reduced
coefficients. In the same fashion, the
{class}`~pymor.reductors.neural_network.NeuralNetworkStatefreeOutputReductor` and the
corresponding {class}`~pymor.models.neural_network.NeuralNetworkStatefreeOutputModel` are extended to
take instationary problems into account.

A slightly different approach that is also implemented in pyMOR and uses a different type of
neural network is described in the following section.

### Long short-term memory neural networks for instationary problems

So-called *recurrent neural networks* are especially well-suited for capturing time-dependent
dynamics. These types of neural networks can treat input sequences of variable length (in our case
sequences with a variable number of time steps) and store internal states that are passed from one
time step to the next. Therefore, these networks implement an internal memory that keeps
information over time. Furthermore, for each element of the input sequence, the same neural
network is applied.

In the {class}`~pymor.models.neural_network.NeuralNetworkModel` obtained by the
{class}`~pymor.reductors.neural_network.NeuralNetworkLSTMReductor`,
we make use of a specific type of recurrent neural network, namely a so-called
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
basis coefficients (similar to the
{class}`~pymor.models.neural_network.NeuralNetworkModel`) or approximations of the
output quantities (similar to the
{class}`~pymor.models.neural_network.NeuralNetworkStatefreeOutputModel`). For state approximations
using a reduced basis, one can apply the
{class}`~pymor.reductors.neural_network.NeuralNetworkLSTMReductor`.
For a direct approximation of outputs using LSTMs, we provide the
{class}`~pymor.reductors.neural_network.NeuralNetworkLSTMStatefreeOutputReductor`.

### Instationary neural network reductors in practice

In the following we apply two neural network reductors to a parametrized parabolic equation.
First, we import the parametrized heat equation example from {mod}`~pymor.models.examples`:

```{code-cell} ipython3
from pymor.models.examples import heat_equation_example
fom = heat_equation_example()
product = fom.h1_0_semi_product
```

We further define the parameter space:

```{code-cell} ipython3
parameter_space = fom.parameters.space(1, 25)
```

Additionally, we sample training, validation and test parameters from the respective parameter space:

```{code-cell} ipython3
training_parameters = parameter_space.sample_uniformly(15)
validation_parameters = parameter_space.sample_randomly(3)
test_parameters = parameter_space.sample_randomly(10)
```

To check how the two reductors perform, we write a simple function that measures the
errors and the speedups on a set of test parameters:

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

We now run the
{class}`~pymor.reductors.neural_network.NeuralNetworkReductor`
and the
{class}`~pymor.reductors.neural_network.NeuralNetworkLSTMReductor`
with different parameters and evaluate their performance:

```{code-cell} ipython3
from pymor.reductors.neural_network import NeuralNetworkReductor, NeuralNetworkLSTMReductor

basis_size = 20

reductor = NeuralNetworkReductor(fom, training_parameters=training_parameters, validation_parameters=validation_parameters, basis_size=basis_size,
                                             pod_params={'product': product}, ann_mse=None, scale_inputs=True,
                                             scale_outputs=True)
rom = reductor.reduce(restarts=0)
rel_errors, speedups = compute_errors(rom, reductor)
reductor_lstm = NeuralNetworkLSTMReductor(fom, training_parameters=training_parameters, validation_parameters=validation_parameters, basis_size=basis_size,
                                                      pod_params={'product': product}, ann_mse=None, scale_inputs=True,
                                                      scale_outputs=True)
rom_lstm = reductor_lstm.reduce(restarts=0, number_layers=1, hidden_dimension=25, learning_rate=0.01)
rel_errors_lstm, speedups_lstm = compute_errors(rom_lstm, reductor_lstm)
```

We finally print the results:

```{code-cell} ipython3
print('Results for the state approximation:')
print('====================================')
print()
print('Approach by Hesthaven and Ubbiali using feedforward ANNs:')
print('---------------------------------------------------------')
print(f'Average relative error: {np.average(rel_errors)}')
print(f'Median of speedup: {np.median(speedups)}')
print()
print('Approach using long short-term memory ANNs:')
print('-------------------------------------------')
print(f'Average relative error: {np.average(rel_errors_lstm)}')
print(f'Median of speedup: {np.median(speedups_lstm)}')
```

In this example, we observe that the LSTMs perform much better than the feedforward ANNs in terms
of accuracy while the speedups of both methods lie in the same order of magnitude.

Download the code:
{download}`tutorial_mor_with_anns.md`
{nb-download}`tutorial_mor_with_anns.ipynb`


## Data-driven neural network without full-order model

In the previous sections, we have seen how to use neural networks for model order reduction 
using a full-order model. However, if there is no full-order model available, one can utilise
the neural network reductor to approximate the mapping from the {{ Parameters }} to the
coefficients of the respective solution in a reduced basis from data which was generated by
an arbitrary external solver. This works for all of the above discussed neural network reductors.

In the following we will show an example with data generated from our initial two-dimensional 
diffusion problem with parametrized diffusion, right hand side and Dirichlet boundary condition.
Therefore, we will use the 
{class}`~pymor.reductors.neural_network.NeuralNetworkStatefreeOutputReductor` to apprximate the
mapping from the {{ Parameters }} to the output quantities directly using the
{class}`~pymor.reductors.neural_network.NeuralNetworkReductor` without the full-order model.

We reuse the `problem` with output dimension {math}`q=2` and create a full order model that
is aware of the output quantities:

```{code-cell} ipython3
fom, _ = discretize_stationary_cg(problem, diameter=1/50)
```

We employ a single {{ Parameter }} with the same range for each
parameter and create the {{ ParameterSpace }}:

```{code-cell} ipython3
parameter_space = fom.parameters.space((0.1, 1))
```

To train the neural network, we create a set of training, validation and test parameters
consisting of 100,20 and 10 randomly chosen {{ parameter_values }}, respectively:

```{code-cell} ipython3
training_parameters = parameter_space.sample_uniformly(100)
validation_parameters = parameter_space.sample_randomly(20)
test_parameters = parameter_space.sample_randomly(10)
```

Contrary to the previous examples, we now generate data from the full order model. Therefore,
we create `output` quantities for the `training_paramters` and `validation_parameters`:

```{code-cell} ipython3
    training_outputs = []
    for mu in training_parameters:
        training_outputs.append(fom.compute(output=True, mu=mu)['output'])
    training_outputs = np.squeeze(np.array(training_outputs))

    validation_outputs = []
    for mu in validation_parameters:
        validation_outputs.append(fom.compute(output=True, mu=mu)['output'])
    validation_outputs = np.squeeze(np.array(validation_outputs))
```

Now we import the {class}`~pymor.reductors.neural_network.NeuralNetworkStatefreeOutputReductor`
and initialize the reductor passing only the {{ Parameters }} and the outputs without the
full order model (`fom`):

```{code-cell} ipython3
from pymor.reductors.neural_network import NeuralNetworkStatefreeOutputReductor

    output_reductor_data_driven = NeuralNetworkStatefreeOutputReductor(training_parameters=training_parameters,
                                                                       training_outputs=training_outputs,
                                                                       validation_parameters=validation_parameters,
                                                                       validation_outputs=validation_outputs,
                                                                       validation_loss=1e-5)
```

Similar to the previous examples, the reduction can be performed using the `reduce` method
and we measure the speed up and the errors on the test parameters of the resulting
{class}`~pymor.models.neural_network.NeuralNetworkStatefreeOutputModel`:

```{code-cell} ipython3
    output_rom_data_driven = output_reductor_data_driven.reduce(restarts=100, log_loss_frequency=10)

    outputs = []
    outputs_red_data_driven = []
    outputs_speedups_data_driven = []
    print(f'Performing test on parameters of size {len(test_parameters)} ...')

    for mu in test_parameters:
        tic = time.perf_counter()
        outputs.append(fom.compute(output=True, mu=mu)['output'])
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red_data_driven.append(output_rom_data_driven.compute(output=True, mu=mu)['output'])
        time_red_data_driven = time.perf_counter() - tic

        outputs_speedups_data_driven.append(time_fom / time_red_data_driven)

    outputs = np.squeeze(np.array(outputs))
    outputs_red_data_driven = np.squeeze(np.array(outputs_red))

    outputs_absolute_errors_data_driven = np.abs(outputs - outputs_red_data_driven)
    outputs_relative_errors_data_driven = np.abs(outputs - outputs_red_data_driven) / np.abs(outputs)
```

The average absolute error (component-wise) on the training parameters is given by

```{code-cell} ipython3
np.average(outputs_absolute_errors_data_driven)
```

The average relative error is

```{code-cell} ipython3
np.average(outputs_relative_errors_data_driven)
```

and the median of the speedups amounts to

```{code-cell} ipython3
np.median(outputs_speedups_data_driven)
```
