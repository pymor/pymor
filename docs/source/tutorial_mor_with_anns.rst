Tutorial 5: Model order reduction with artificial neural networks
=================================================================

.. code-links::
    :timeout: -1


Recent success of artificial neural networks led to the development of several
methods for model order reduction using neural networks. pyMOR provides the
functionality for an approach developed by Hesthaven and Ubbiali in 2018.
pyMOR serves as discretization tool and for computing a reduced basis via
proper orthogonal decomposition. Further, the PyTorch library is used to train
artificial neural networks.

In this tutorial we will learn about the basic idea of the approach and how to
use it in pyMOR.

We assume that we are given a parametrized partial differential equation which
can be discretized and solved using pyMOR. In this example, we consider the
following diffusion problem on a one dimensional line domain with a
parametrized diffusion:
.. math::

   -\nabla \cdot \big(\sigma(x, \mu) \nabla u(x, \mu) \big) = f(x, \mu),\quad x \in \Omega,

on the domain :math:`\Omega:= (0, 1) \subset \mathbb{R}` with data
functions :math:`f(x, \mu) = 1000 \cdot (x-0.5)^2`,
:math:`\sigma(x, \mu)=(1-x)*\mu+x`, where :math:`\mu \in (0.1, 1)` denotes the
parameter. Further, we apply homogeneous Dirichlet boundary conditions.
We discretize the problem as explained in former tutorials:
.. nbplot::

    from pymor.basic import *
    
    domain = LineDomain()

    N = 100

    f = ExpressionFunction('(x - 0.5)**2 * 1000', 1, ())

    d0 = ExpressionFunction('1 - x', 1, ())
    d1 = ExpressionFunction('x', 1, ())

    v0 = ProjectionParameterFunctional('mu')
    v1 = 1.

    diffusion = LincombFunction([d0, d1], [v0, v1])

    problem = StationaryProblem(
        domain=domain,
        rhs=rhs,
        diffusion=diffusion,
        dirichlet_data=ConstantFunction(value=0, dim_domain=1),
        name='1DProblem'
    )

    fom, _ = discretize_stationary_cg(problem, diameter=1. / N))

Since we employ a single |Parameter|, we can create the |ParameterSpace| using
the following line:
.. nbplot::

    parameter_space = fom.parameters.space((0.1, 1))

The main idea of the approach by Hesthaven et al. is to approximate the mapping
from the |Parameters| to the coefficients of the respective solution in a
reduced basis by means of a neural network. Thus, in the online phase, one
performs a forward pass of the |Parameters| through the neural networks and
obtains the approximated reduced coordinates. To derive the corresponding
high-fidelity solution, one can further use the reduced basis and compute the
linear combination defined by the reduced coefficients. The reduced basis is
created via POD.

To train the neural network, we create a training and a validation set
consisting of 100 and 20 randomly chosen |Parameters|, respectively:
.. nbplot::

    training_set = parameter_space.sample_uniformly(100)
    validation_set = parameter_space.sample_randomly(20)

In this tutorial, we prescribe the size of the reduced basis that shall be
used. It is also possible to determine a relative or absolute tolerance that
should not be exceeded on the validation set. We can now construct a reductor
using a basis size of 10:
.. nbplots::

    from pymor.reductors.neural_network import NeuralNetworkReductor

    reductor = NeuralNetworkReductor(fom, training_set, validation_set, basis_size=10)

To reduce the model, i.e. compute a reduced basis via POD and train the neural
network, we use the respective function of the
:class:`~pymor.reductors.neural_network.NeuralNetworkReductor`:
.. nbplot::

    rom = reductor.reduce()

This function will automatically train several neural networks with different
initial weights and select the one leading to the best results on the
validation set.

We are now ready to test our implementation by solving for a random parameter
the full problem and the reduced model and visualize the result:
.. nbplot::

    mu = parameter_space.sample_randomly(1)[0]

    U = fom.solve(mu)
    U_red = rom.solve(mu)
    U_red_recon = reductor.reconstruct(U_red)

    fom.visualize((U, U_red_recon),
                  legend=(f'Full solution for mu={mu}', f'Reduced solution for mu={mu}'))
