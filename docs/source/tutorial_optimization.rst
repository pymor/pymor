Tutorial: Model order reduction for PDE-constrained optimization problems
=========================================================================

.. include:: jupyter_init.txt

A typical application of model order reduction for PDEs are
PDE-constrained parameter optimization problems. These problems aim to
find a local minimizer of an objective functional which underlies a PDE
that needs to be solved for all evaluations. Mathematically speaking,
for a physical domain :math:`\Omega \subset \mathbb{R}^d` and a
parameter set :math:`\mathcal{P} \subset \mathbb{R}^P`, we want to find
a local minimizer :math:`\mu \in \mathcal{P}` of

.. math::


   \min_{\mu \in \mathcal{P}} J(u_{\mu}, \mu),  \tag{P.a}

 where :math:`u_{\mu} \in V := H^1_0(\Omega)` is the solution of

.. math::

   \begin{equation} \label{eq:primal}
   a_{\mu}(u_{\mu}, v) = f_{\mu}(v) \qquad \forall \, v \in V \tag{P.b}.
   \end{equation}

The equation :raw-latex:`\eqref{eq:primal}` is called the primal
equation and can be arbitrariry complex. MOR methods in the context of
PDE-constrained optimization problems thus aim to find a surrogate model
of :raw-latex:`\eqref{eq:primal}` to reduce the computational costs of
an evaluation of :math:`J(u_{\mu}, \mu)`.

Since :math:`u_{\mu}` is always related to :math:`\mu`, we can also
simplify (P) by using the so-called reduced objective functional
:math:`\mathcal{J}(\mu):= J(u_{\mu}, \mu)`. Note that 'reduced' is not
referring to the order of the model. We get the reduced optimization
problem: Find a local minimizer of

.. math::

    
   \min_{\mu \in \mathcal{P}} \mathcal{J}(\mu),  \tag{$\hat{P}$}

There exist plenty of different methods to solve (:math:`\hat{P}`) by
using MOR methods. Some of them rely on an RB method with traditional
offline/online splitting, which typically result in a very online
efficient approach. Recent reasearch also tackles overall efficiency by
overcoming the expensive offline approach which we will discuss further
below.

In this tutorial, we use a simple linear objective functional and an
elliptic primal equation to compare different approaches to solve
(:math:`\hat{P}`). For a more advanced problem class, we refer to `this
project <https://github.com/TiKeil/NCD-corrected-TR-RB-approach-for-pde-opt>`__.

An elliptic model problem with a linear objective functional
------------------------------------------------------------

We consider a domain :math:`\Omega:= [-1, 1]^2`, a parameter set
:math:`\mathcal{P} := [0,\pi]^2` and the elliptic equation

.. math::


   - \nabla \cdot \big( \lambda_\mu \nabla u_\mu \big) = l

 with data functions

.. raw:: latex

   \begin{align}
   l(x, y) &= \tfrac{1}{2} \pi^2 cos(\tfrac{1}{2} \pi x) cos(\tfrac{1}{2} \pi y),\\
   \lambda(\mu) &= \theta_0(\mu) \lambda_0 + \theta_1(\mu) \lambda_1,\\
   \theta_0(\mu) &= 1.1 + sin(\mu_0)\mu_1,\\
   \theta_1(\mu) &= 1.1 + sin(\mu_1),\\
   \lambda_0 &= \chi_{\Omega \backslash \omega},\\
   \lambda_1 &= \chi_\omega,\\
   \omega &:= [-\tfrac{2}{3}, -\tfrac{1}{3}]^2 \cup ([-\tfrac{2}{3}, -\tfrac{1}{3}] \times [\tfrac{1}{3}, \tfrac{2}{3}]).
   \end{align}

The diffusion is thus given as the linear combination of scaled
indicator functions with support inside two blocks :math:`\omega` in the
left half of the domain, roughly where the ``w`` is here:

::

    +-----------+
    |           |
    |  w        |
    |           |
    |  w        |
    |           |
    +-----------+

From the definition above we can easily deduce the bilinear form
:math:`a_{\mu}` and the linear functional :math:`f_{\mu}` for the primal
equation. Moreover, we consider the linear objective functional

.. math::


   \mathcal{J}(\mu) := \theta_{\mathcal{J}}(\mu)\, f_\mu(u_\mu) 

 where
:math:`\theta_{\mathcal{J}}(\mu) := 1 + \frac{1}{5}(\mu_0 + \mu_1)`.

With this data, we can build a \|StationaryProblem\| in pyMOR.

.. jupyter-execute::

    domain = RectDomain(([-1,-1], [1,1]))
    indicator_domain = ExpressionFunction(
        '(-2/3. <= x[..., 0]) * (x[..., 0] <= -1/3.) * (-2/3. <= x[..., 1]) * (x[..., 1] <= -1/3.) * 1. \
       + (-2/3. <= x[..., 0]) * (x[..., 0] <= -1/3.) *  (1/3. <= x[..., 1]) * (x[..., 1] <=  2/3.) * 1.', 
        dim_domain=2, shape_range=())
    rest_of_domain = ConstantFunction(1, 2) - indicator_domain
    
    f = ExpressionFunction('0.5*pi*pi*cos(0.5*pi*x[..., 0])*cos(0.5*pi*x[..., 1])', dim_domain=2, shape_range=())
    
    parameters = {'diffusion': 2}
    thetas = [ExpressionParameterFunctional('1.1 + sin(diffusion[0])*diffusion[1]', parameters,
                                           derivative_expressions={'diffusion': ['cos(diffusion[0])*diffusion[1]',
                                                                                 'sin(diffusion[0])']}),
              ExpressionParameterFunctional('1.1 + sin(diffusion[1])', parameters,
                                           derivative_expressions={'diffusion': ['0',
                                                                                 'cos(diffusion[1])']}),
    
                                           ]
    diffusion = LincombFunction([rest_of_domain, indicator_domain], thetas)
    
    theta_J = ExpressionParameterFunctional('1 + 1/5 * diffusion[0] + 1/5 * diffusion[1]', parameters,
                                               derivative_expressions={'diffusion': ['1/5','1/5']})
    
    problem = StationaryProblem(domain, f, diffusion, outputs=[('l2', f * theta_J)])

pyMOR supports to choose an output function in the
\|StationaryProblem\|. So far :math:`L^2` and :math:`L^2`-boundary
integrals over :math:`u_\mu`, multiplied by an arbitrary \|Function\|,
are supported. These outputs can also be handled by the discretizer. In
our case, we make use of the :math:`L^2` output and multiply it by the
term :math:`\theta_\mu l_\mu`.

We now use the standard builtin discretization tool (see tutorial) to
get a full order \|StationaryModel\|. Since we intend to use a fixed
energy norm

.. math:: \|\,.\|_{\bar{\mu}} : = a_{\,\bar{\mu}}(.,.),

we also define :math:`\bar{\mu}`, which we parse via the argument
``mu_energy_product``. Also, we define the parameter space
:math:`\mathcal{P}` on which we want to optimize.

.. jupyter-execute::

    mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])
    
    fom, data = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)
    parameter_space = fom.parameters.space(0, np.pi)




In case, you need an output functional that can not be defined in the
\|StationaryProblem\|, you can also directly define the
``output_functional`` in the \|StationaryModel\|.

.. jupyter-execute::

    output_functional = fom.rhs.H * theta_J 
    fom = fom.with_(output_functional=output_functional)

To overcome that pyMORs outputs return a \|NumpyVectorArray\|, we have
to define a function that returns numpy arrays instead.

.. jupyter-execute::

    def fom_objective_functional(mu):
        return fom.output(mu).to_numpy()

Of course, all optimization methods need a certain starting parameter,
which in our case is :math:`\mu_0 = (0.25,0.5)`.

.. jupyter-execute::

    initial_guess = fom.parameters.parse([0.25, 0.5])

Next, we visualize the diffusion function :math:`\lambda_\mu` by using
\|InterpolationOperator\| for interpolating it on the grid.

.. jupyter-execute::

    from pymor.discretizers.builtin.cg import InterpolationOperator
    
    diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(initial_guess)
    fom.visualize(diff)



.. jupyter-execute::

    print(data['grid'])


We can see that our FOM model has 20201 DoFs which just about suffices
to resolve the data structure in the diffusion. This suggests to use an
even finer mesh. However, for enabling a faster runtime for this
tutorial, we stick with this mesh and remark that refining the mesh does
not change the interpretation of the methods that are discussed below.
It rather aggravates the result.

Before we start the first optimization method, we define helpful
functions for visualizations.

.. jupyter-execute::

    def compute_value_matrix(f, x, y):
        f_of_x = np.zeros((len(x), len(y)))
        for ii in range(len(x)):
            for jj in range(len(y)):
                f_of_x[ii][jj] = f((x[ii], y[jj]))
        x, y = np.meshgrid(x, y)
        return x, y, f_of_x
    
    def plot_3d_surface(f, x, y, alpha=1):
        X, Y = x, y
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x, y, f_of_x = compute_value_matrix(f, x, y)
        ax.plot_surface(x, y, f_of_x, cmap='Blues',
                        linewidth=0, antialiased=False, alpha=alpha)
        ax.view_init(elev=27.7597402597, azim=-39.6370967742)
        ax.set_xlim3d([-0.10457963, 3.2961723])
        ax.set_ylim3d([-0.10457963, 3.29617229])
        return ax
    
    def addplot_xy_point_as_bar(ax, x, y, color='orange', z_range=None):
        ax.plot([y, y], [x, x], z_range if z_range else ax.get_zlim(), color)

Now, we can visualize the objective functional on the parameter space

.. jupyter-execute::

    logger.info('plotting ...')
    
    ranges = parameter_space.ranges['diffusion']
    XX = np.linspace(ranges[0] + 0.05, ranges[1], 10)
    YY = XX
    
    plot_3d_surface(fom_objective_functional, XX, YY)
    
    logger.info('... done')



Taking a closer look at the functional, we see that it is at least
locally convex with a locally unique minimum. In general, however,
PDE-constrained optimization problems are not convex. In our case
changing the parameter functional :math:`\theta_{\mathcal{J}}` can
already result in a very different non-convex output functional.

In order to record some data in the optimization, we also define two
helpful functions for recording and reporting the results.

.. jupyter-execute::

    reference_minimization_data = {'num_evals': 0,
                                   'evaluations' : [],
                                   'evaluation_points': [],
                                   'time': np.inf}
    
    def record_results(function, data, mu):
        QoI = function(mu)
        data['num_evals'] += 1
        # we need to make sure to copy the data, since the added mu will be changed inplace by minimize afterwards
        data['evaluation_points'].append([fom.parameters.parse(mu)['diffusion'][:][0], 
                                          fom.parameters.parse(mu)['diffusion'][:][1]])
        data['evaluations'].append(QoI[0])
        print('.', end='')
        return QoI
    
    def report(result, data, reference_mu=None):
        if (result.status != 0):
            print('\n failed!')
        else:
            print('\n succeded!')
            print('  mu_min:    {}'.format(fom.parameters.parse(result.x)))
            print('  J(mu_min): {}'.format(result.fun[0]))
            if reference_mu is not None:
                print('  absolute error w.r.t. reference solution: {:.2e}'.format(np.linalg.norm(result.x-reference_mu)))
            print('  num iterations:     {}'.format(result.nit))
            print('  num function calls: {}'.format(data['num_evals']))
            print('  time: {:.5f} seconds'.format(data['time']))
            if 'offline_time' in data:
                print('  offline time: {:.5f} seconds'.format(data['offline_time']))
            if 'enrichments' in data:
                print('  model enrichments: {}'.format(data['enrichments']))
        print('')

Optimizing with the FOM using finite differences
------------------------------------------------

There exist plenty optimization methods and this tutorial is not meant
to discuss the design and implementation of optimization methods. We
simply use the ``minimize`` function from ``scipy.optimize`` and use the
builtin ``L-BFGS-B`` routine which is a quasi-Newton method that can
also handle a constrained parameter space.

It is optional to give an expression for the gradient of the objective
functional to the ``minimize`` function. In case no gradient is given,
``minimize`` just approximates the gradient with finite differences.
This is not recommended because the gradient is inexact and the
computation of finite difference requires even more evalutations of the
primal equation. We anyway start with this approach.

.. jupyter-execute::

    from functools import partial
    from scipy.optimize import minimize
    
    tic = perf_counter()
    fom_result = minimize(partial(record_results, fom_objective_functional, reference_minimization_data),
                          initial_guess.to_numpy(),
                          method='L-BFGS-B', jac=False,
                          bounds=(ranges, ranges),
                          options={'ftol': 1e-15})
    reference_minimization_data['time'] = perf_counter()-tic
    reference_mu = fom_result.x



.. jupyter-execute::

    report(fom_result, reference_minimization_data)


taking a look at the result, we see that the optimizer needs :math:`9`
iterations to converge, but actually needs :math:`48` evaluations of the
full order model. Obiously, this is related to the computation of the
finite differences. We can visualize the optimization path by plotting
the chosen points during the minimization.

.. jupyter-execute::

    reference_plot = plot_3d_surface(fom_objective_functional, XX, YY, alpha=0.5)
    
    for mu in reference_minimization_data['evaluation_points']:
        addplot_xy_point_as_bar(reference_plot, mu[0], mu[1])




Optimizing with the ROM using finite differences
------------------------------------------------

We can use a standard RB method to build a surrogate model for the FOM.
As a result, the solution of the primal equation is no longer expensive
and the optimization method has the chance to converge faster. For this,
we define a standard ``reductor`` and use the
``MinThetaParameterFunctional`` for an estimation of the coerciviy
constant.

.. jupyter-execute::

    from pymor.algorithms.greedy import rb_greedy
    from pymor.reductors.coercive import CoerciveRBReductor
    
    from pymor.parameters.functionals import MinThetaParameterFunctional
    
    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)
    
    training_set = parameter_space.sample_uniformly(25)
    training_set_simple = [mu['diffusion'] for mu in training_set]
    
    RB_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)



Using MOR for PDE-constrained optimization goes beyond the classical
online efficiency of RB methods. It is not meaningful to ignore the
offline time of the surrogate model since it can happen that FOM
optimization methods would already converge before the surrogate model
is even ready. Thus, the RB optimization methods (at least for only one
configuration) aims for overall efficiency which includes offline and
online time. Of course, this effect aggravates if the parameter space is
high dimensional.

In order to decrease the offline time we realize that we do not require
a perfect surrogate model in the sense that a low error tolerance for
the ``rb_greedy`` already suffices to converge to the same minimum. In
our case we choose ``atol=1e-2`` and yield a very low dimensional space.
In general, however, it is not a priorily clear how to choose ``atol``
in order to arrive at a minimum which is close enough to the true
optimum.

.. jupyter-execute::

    RB_greedy_data = rb_greedy(fom, RB_reductor, training_set, atol=1e-2)
    
    num_RB_greedy_extensions = RB_greedy_data['extensions']
    RB_greedy_mus, RB_greedy_errors = RB_greedy_data['max_err_mus'], RB_greedy_data['max_errs']
    
    rom = RB_greedy_data['rom']
    
    logger.info('RB system is of size {}x{}'.format(num_RB_greedy_extensions, num_RB_greedy_extensions))
    logger.info('maximum estimated model reduction error over training set: {}'.format(RB_greedy_errors[-1]))

As we can see the greedy already stops after :math:`3` basis functions.
Next, we plot the chosen parameters.

.. jupyter-execute::

    ax = plot_3d_surface(fom_objective_functional, XX, YY, alpha=0.5)
    
    for mu in RB_greedy_mus[:-1]:
        mu = mu.to_numpy()
        addplot_xy_point_as_bar(ax, mu[0], mu[1])





Analogously to above, we perform the same optimization method, but use
the resulting ROM objective functional.

.. jupyter-execute::

    def rom_objective_functional(mu):
        return rom.output(mu).to_numpy()
    
    RB_minimization_data = {'num_evals': 0,
                            'evaluations' : [],
                            'evaluation_points': [],
                            'time': np.inf,
                            'offline_time': RB_greedy_data['time']
                            }
    
    tic = perf_counter()
    rom_result = minimize(partial(record_results, rom_objective_functional, RB_minimization_data),
                          initial_guess.to_numpy(),
                          method='L-BFGS-B', jac=False,
                          bounds=(ranges, ranges),
                          options={'ftol': 1e-15})
    RB_minimization_data['time'] = perf_counter()-tic


.. jupyter-execute::

    report(rom_result, RB_minimization_data, reference_mu)

Comparing the result to the FOM model, we see that the number of
iterations and evaluations of the model slightly decreased. As expected,
we see that the optmization routine is very fast because the surrogate
enables almost instant evaluations of the primal equation.

As mentioned above, we should not forget that we required the offline
time to build our surrogate. In our case, the offline time is still low
enough to get a speed up over the FOM optimization. Luckily,
``atol=1e-2`` was enough to achieve an absolute error of ``1.35e-06``
but it is important to notice that we do not know this error before
choosing ``atol``.

To show that the ROM optimization roughly followed the same path as the
FOM optimization, we visualize both of them in the following plot.

.. jupyter-execute::

    reference_plot = plot_3d_surface(fom_objective_functional, XX, YY, alpha=0.5)
    reference_plot_mean_z_lim = 0.5*(reference_plot.get_zlim()[0] + reference_plot.get_zlim()[1])
    
    for mu in reference_minimization_data['evaluation_points']:
        addplot_xy_point_as_bar(reference_plot, mu[0], mu[1], color='green',
                                z_range=(reference_plot.get_zlim()[0], reference_plot_mean_z_lim))
    
    for mu in RB_minimization_data['evaluation_points']:
        addplot_xy_point_as_bar(reference_plot, mu[0], mu[1], color='orange',
                               z_range=(reference_plot_mean_z_lim, reference_plot.get_zlim()[1]))




.. image:: output_44_1.png


Computing the gradient of the objective functional
--------------------------------------------------

A major issue of using finite differences for computing the gradient of
the objective functional is the number of evaluations of the objective
functional. In the FOM example from above we saw that
:math:`48 - 9 = 39` evaluations of the model were only due to the
computation of the finite differences. If the problem is more complex
and the mesh is finer, this can lead us into a serious waste of
computational time. Also from an optimizational point of view it is
always better to compute the true gradient of the objective functional.

For computing the gradient of the linear objective functional
:math:`\mathcal{J}(\mu)`, we can write for every direction
:math:`i= 1, \dots, P`

.. math::


   \begin{align} \label{gradient:sens} \tag{1}
   d_{\mu_i} \mathcal{J}(\mu) = \partial_{\mu_i} J(u_{\mu}, \mu) + \partial_u J(u_{\mu}, \mu)[d_{\mu_i} u_{\mu}] 
      =   \partial_{\mu_i} J(u_{\mu}, \mu) + J(d_{\mu_i} u_{\mu}, \mu)
   \end{align}

where :math:`d` means total derivative and :math:`\partial` means
partial derivative. Thus, we need to compute the derivative of the
solution :math:`u_{\mu}` (also called sensitivity). For this, we need to
solve another equation: Find :math:`d_{\mu_i} u_{\mu} \in V`, such that

.. math::

    \label{sens} \tag{2}
   a_\mu(d_{\mu_i} u_{\mu}, v) = \partial_{\mu_i} r_\mu^{\text{pr}}(u_{\mu})[v] \qquad \qquad \forall v \in V

where :math:`r_\mu^{\text{pr}}` denotes the residual of the primal
equation. A major issue of this approach is that the computation of the
full gradient requires :math:`P` solutions of :raw-latex:`\eqref{sens}`.
Especially for high dimensional parameter spaces, we can instead use the
adjoint approach to reduce the computational cost to only one solution
of an additional problem.

The adjoint approach relies on the Lagrangian of the objective
functional

.. math::


   \mathcal{L}(u, \mu, p) = J(u, \mu) + r_\mu^{\text{pr}}(u, p)

where :math:`p \in V` is the adjoint variable. Deriving optimality
conditions for :math:`\mathcal{L}`, we end up with the dual equation:
Find :math:`p_{\mu} \in V`, such that

.. math::

    \label{dual} \tag{3}
   a_\mu(v, p_\mu) = \partial_u J(u_\mu, \mu)[v]
   = J(v, \mu)

Note that in our case, we then have
:math:`\mathcal{L}(u_{\mu}, \mu, p_{\mu}) = J(u, \mu)` because the
residual term :math:`r_\mu^{\text{pr}}(u_{\mu}, p_{\mu})` vanishes. By
using the dual problem, we can then derive the gradient of the objective
functional by

.. math::


   \begin{align} 
   d_{\mu_i} \mathcal{J}(\mu) &= \partial_{\mu_i} J(u_{\mu}, \mu) + \partial_u J(u_{\mu}, \mu)[d_{\mu_i} u_{\mu}] \\ 
      &=   \partial_{\mu_i} J(u_{\mu}, \mu) + a_\mu(d_{\mu_i} u_{\mu}, p_\mu) \\
      &=   \partial_{\mu_i} J(u_{\mu}, \mu) + \partial_{\mu_i} r_\mu^{\text{pr}}(d_{\mu_i} u_{\mu})[p_\mu]
   \end{align}

We conclude that we only need to solve for :math:`u_{\mu}` and
:math:`p_{\mu}` if we want to compute the gradient with the adjoint
approach.

We now intend to use the gradient to speed up the optimization methods
from above. All technical requirements are
already available in pyMOR.


Optimizing using a gradient in FOM
----------------------------------

We can easily include a function to compute the gradient to ``minimize``.


.. jupyter-execute::

    def fom_gradient_of_functional(mu):
        return fom.output_functional_gradient(opt_fom.parameters.parse(mu))
    
    opt_fom_minimization_data = {'num_evals': 0,
                                'evaluations' : [],
                                'evaluation_points': [],
                                'time': np.inf}
    tic = perf_counter()
    opt_fom_result = minimize(partial(record_results, fom_objective_functional, opt_fom_minimization_data),
                              initial_guess.to_numpy(),
                              method='L-BFGS-B', 
                              jac=fom_gradient_of_functional,
                              bounds=(ranges, ranges),
                              options={'ftol': 1e-15})
    opt_fom_minimization_data['time'] = perf_counter()-tic
    
    # update the reference_mu because this is more accurate!
    reference_mu = opt_fom_result.x




.. jupyter-execute::

    report(opt_fom_result, opt_fom_minimization_data)

    


With respect to the FOM result with finite differences, we see that we
have a massive speed up by computing the gradient information properly.

Optimizing using a gradient in ROM
----------------------------------

Obviously, we can also include the gradient of the ROM version of the
output functional.


.. jupyter-execute::

    def rom_gradient_of_functional(mu):
        return rom.output_functional_gradient(opt_rom.parameters.parse(mu))
                
    
    opt_rom_minimization_data = {'num_evals': 0,
                                 'evaluations' : [],
                                 'evaluation_points': [],
                                 'time': np.inf,
                                 'offline_time': RB_greedy_data['time']}
    
    
    tic = perf_counter()
    opt_rom_result = minimize(partial(record_results, rom_objective_functional, opt_rom_minimization_data),
                      initial_guess.to_numpy(),
                      method='L-BFGS-B', 
                      jac=rom_gradient_of_functional,
                      bounds=(ranges, ranges),
                      options={'ftol': 1e-15})
    opt_rom_minimization_data['time'] = perf_counter()-tic
    report(opt_rom_result, opt_rom_minimization_data, reference_mu)

    


The online phase is even slightly faster than before but the offline
phase is obviously still the same as before. We also conclude that the
ROM model eventually gives less speedup by using a better optimization
method for the FOM and ROM. This is something that is a common issue for
MOR methods in the context of optimization.

Breaking the traditional offline/online splitting: enrich along the path of optimization
----------------------------------------------------------------------------------------

We already figured that the main drawback for using RB methods in the
context of optimization is the expensive offline time to build the
surrogate model. In the example above, we overcame this issue by
choosing a very high tolerance ``atol``. As a result, we can not be sure
that our surrogate model is accurate enough for our purpuses. In other
words, either we invest too much time to build an accurate model or we
face the danger of reducing with a bad surrogate for the whole parameter
space. Thinking about this issue again, it is important to notice that
we are solving an optimization problem which will eventually converge to
a certain parameter. Thus, it only matters that the surrogate is good in
this particular reason as long as we are able to arrive at it. This
gives hope that there must exists a more efficient way of using RB
methods without trying to approximate the whole parameter space.

One possible way for advanced RB methods is a reduction along the path
of optimization. The idea is, that we start with an empty basis and only
enrich the model with the parameters that we will arive at. This
approach goes beyond the classical offline/online splitting of RB
methods since it entirely skips the offline phase. In the following
code, we will test this method.



.. jupyter-execute::

    pdeopt_reductor = StationaryCoerciveRBReductor(
        fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)

In the next function, we implement the above mentioned way of enriching
the basis along the path of optimization.

.. jupyter-execute::

    def record_results_and_enrich(function, data, opt_dict, mu):
        U = fom.solve(mu)
        try:
            pdeopt_reductor.extend_basis(U)
            data['enrichments'] += 1
        except:
            logger.info('Extension failed')
        opt_rom = pdeopt_reductor.reduce()
        QoI = rom.output(mu).to_numpy()
        data['num_evals'] += 1
        # we need to make sure to copy the data, since the added mu will be changed inplace by minimize afterwards
        data['evaluation_points'].append([fom.parameters.parse(mu)['diffusion'][:][0], 
                                          fom.parameters.parse(mu)['diffusion'][:][1]])
        data['evaluations'].append(QoI[0])
        opt_dict['opt_rom'] = rom 
        print('.', end='')
        return QoI
    
    def compute_gradient_with_opt_rom(opt_dict, mu):
        rom = opt_dict['opt_rom']
        return opt_rom.output_functional_gradient(rom.parameters.parse(mu))

.. jupyter-execute::

    opt_along_path_minimization_data = {'num_evals': 0,
                                           'evaluations' : [],
                                           'evaluation_points': [],
                                           'time': np.inf,
                                           'enrichments': 0}
    opt_dict = {} 
    tic = perf_counter()
    opt_along_path_result = minimize(partial(record_results_and_enrich, rom_objective_functional, 
                                             opt_along_path_minimization_data, opt_dict),
                                      initial_guess.to_numpy(),
                                      method='L-BFGS-B', 
                                      jac=partial(compute_gradient_with_opt_rom, opt_dict),
                                      bounds=(ranges, ranges),
                                      options={'ftol': 1e-15})
    opt_along_path_minimization_data['time'] = perf_counter()-tic



.. jupyter-execute::

    report(opt_along_path_result, opt_along_path_minimization_data, reference_mu)

    


The computational time looks at least better than the FOM optimization
and we are very close to the reference parameter saved some
computational time. But we are following the exact same path than the
FOM and thus we need to solve the FOM model as often as before. The only
time that we safe is the one for the dual solution which we compute with
the ROM instead.

Adaptively enriching along the path
-----------------------------------

This makes us think about another idea where we only enrich if it is
neccessary. For example it could be that the model is already good at
the next iteration, which we can easily check by evaluating the standard
error estimator which is also used in the greedy algorithm. In the next
example we will implement this adaptive way of enriching and set a
tolerance which is equal to the one that we had in the construction of
the greedy.

.. jupyter-execute::

    pdeopt_reductor = StationaryCoerciveRBReductor(
        opt_fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
    opt_rom = pdeopt_reductor.reduce()




.. jupyter-execute::

    def record_results_and_enrich_adaptively(function, data, opt_dict, mu):
        opt_rom = opt_dict['opt_rom'] 
        primal_estimate = opt_rom.estimate_error(opt_rom.solve(mu), opt_rom.parameters.parse(mu))
        if primal_estimate > 1e-2:
            logger.info('Enriching the space because primal estimate is {} ...'.format(primal_estimate))
            U = opt_fom.solve(mu)
            try:
                pdeopt_reductor.extend_basis(U)
                data['enrichments'] += 1
                opt_rom = pdeopt_reductor.reduce()
            except:
                logger.info('... Extension failed')
        else:
            logger.info('Do NOT enrich the space because primal estimate is {} ...'.format(primal_estimate))
        opt_rom = pdeopt_reductor.reduce()
        QoI = opt_rom.output(mu).to_numpy()
        data['num_evals'] += 1
        # we need to make sure to copy the data, since the added mu will be changed inplace by minimize afterwards
        data['evaluation_points'].append([fom.parameters.parse(mu)['diffusion'][:][0], 
                                          fom.parameters.parse(mu)['diffusion'][:][1]])
        data['evaluations'].append(QoI[0])
        opt_dict['opt_rom'] = opt_rom 
        print('.', end='')
        return QoI
    
    def compute_gradient_with_opt_rom(opt_dict, mu):
        opt_rom = opt_dict['opt_rom']
        return opt_rom.output_functional_gradient(opt_rom.parameters.parse(mu))

.. jupyter-execute::

    opt_along_path_adaptively_minimization_data = {'num_evals': 0,
                                           'evaluations' : [],
                                           'evaluation_points': [],
                                           'time': np.inf,
                                           'enrichments': 0}
    opt_dict = {'opt_rom': opt_rom}
    tic = perf_counter()
    opt_along_path_adaptively_result = minimize(partial(record_results_and_enrich_adaptively, rom_objective_functional, 
                                                        opt_along_path_adaptively_minimization_data, opt_dict),
                                                initial_guess.to_numpy(),
                                                method='L-BFGS-B', 
                                                jac=partial(compute_gradient_with_opt_rom, opt_dict),
                                                bounds=(ranges, ranges),
                                                options={'ftol': 1e-15})
    opt_along_path_adaptively_minimization_data['time'] = perf_counter()-tic



.. jupyter-execute::

    report(opt_along_path_adaptively_result, opt_along_path_adaptively_minimization_data, reference_mu)

    


Now, we actually only needed :math:`4` enrichments and ended up with an
approximation error of ``3.48e-07`` while getting the highest speed up
amongst all methods that we have seen above. To conclude, we once again
compare all methods that we have discussed in this notebook.

.. jupyter-execute::

    print('FOM with finite differences')
    report(fom_result, reference_minimization_data, reference_mu)
    
    print('\nROM with finite differences')
    report(rom_result, RB_minimization_data, reference_mu)
    
    print('\nFOM with gradient')
    report(opt_fom_result, opt_fom_minimization_data, reference_mu)
    
    print('\nROM with gradient')
    report(opt_rom_result, opt_rom_minimization_data, reference_mu)
    
    print('\nAlways enrich along the path')
    report(opt_along_path_result, opt_along_path_minimization_data, reference_mu)
    
    print('\nAdaptively enrich along the path')
    report(opt_along_path_adaptively_result, opt_along_path_adaptively_minimization_data, reference_mu)


    

Some general words about MOR methods for optimization
-----------------------------------------------------

This notebook has shown several aspects on how to use RB methods for
reducing a FOM for an optimization problem. One main result from this
was that standard RB methods can help to reduce the computational time.
Thus, standard RB methods are especially of interest if an optimization
problem might need to solved multiple times.

However, for only a single optimization routine, their expensive offline
time might make them unfavorable because they lack overall efficiency.
The example that has been discussed in this notebook is a very simple
and low dimensional problem with a linear output functional. Especially
going to high dimensional parameter spaces and non linear output
functionals would aggravate this effect even more.

To resolve this issue we have seen a way to overcome the traditional
offline/online splitting and saw that it is a good idea to enrich the
model along the path of the optimization or (even better) only enrich
the model if the standard error estimator goes above a certain
tolerance.

Furthermore, higher order optmization methods with accessible gradient
or hessian make FOM methods take even less steps. Also in this case,
adaptive RB methods still reduce the computational demand of the
optimization method.

For more advanced methods and problems on this topic, we refer to
Trust-Region methods, quadratic objective functionals, inverse problems
or higher order optimization methods. A Trust-Region method for
quadratic objective functionals with a non-conforming RB approach has
been considered in `this (arXiv)
paper <https://arxiv.org/abs/2006.09297>`__, where pyMOR has been used
for the MOR part. You can see the whole code and all numerical results
in `this
project <https://github.com/TiKeil/NCD-corrected-TR-RB-approach-for-pde-opt>`__.

Download the code:
:jupyter-download:script:`tutorial_optimization`
:jupyter-download:notebook:`tutorial_optimization`
