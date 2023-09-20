from pymor.basic import *
from functools import partial
import numpy as np
from time import perf_counter
import trust_region_method
#from quadratic_minimize_functional import assemble_l2_product_functional
from scipy.sparse import coo_matrix, csc_matrix, dia_matrix
from pymor.algorithms.newton import newton
from discretize_cg_with_nonlinear_reactionoperator import discretize_stationary_cg as discretizer
from discretize_cg_with_nonlinear_reactionoperator import element_NonlinearReactionOperator
from discretize_cg_with_nonlinear_reactionoperator import quadratic_functional, element_quadratic_functional
import stationary_problem
from scipy.optimize import linprog


def constraints_A(grid, u_ROM, lam_ROM, d_mu_1_u_ROM, d_mu_2_u_ROM, NonlinearReactionOperator_e, mu, basis):

    C_dv_ub = (-grid.volumes(0)).tolist()
    C_dv_lb = grid.volumes(0).tolist()
    C_rp_ub = []
    C_rp_lb = []
    C_ra_ub = []
    C_ra_lb = []
    C_ga_ub = []
    C_ga_lb = []
    C_q_ub = []
    C_q_lb = []
    C_rs_ub_1 = []
    C_rs_lb_1 = []
    C_rs_ub_2 = []
    C_rs_lb_2 = []
    NonlinearReactionOperator = fom.operator.operators[4]
    _1, element_contribution_NonlinearReactionOperator = NonlinearReactionOperator.apply(basis.lincomb(u_ROM.to_numpy()), element_contribution = True)
    _2, element_contribution_jacobian_NonlinearReactionOperator = NonlinearReactionOperator.jacobian(basis.lincomb(u_ROM.to_numpy()), element_contribution = True)
    _3, element_contribution_j = j.apply(basis.lincomb(u_ROM.to_numpy()), mu = mu, element_contribution = True)
    _4, element_contribution_jacobian_j = j.jacobian(basis.lincomb(u_ROM.to_numpy()), element_contribution = True)
    _5, element_contribution_j_d_mu = j.d_mu(mu_k, element_contribution = True)
    #rho = np.zeros(grid.size(0))
    for e in range(grid.size(0)):

        reduced_residual_e = project(NumpyMatrixOperator(element_contribution_NonlinearReactionOperator[e].to_numpy().T,
                                            range_id = NonlinearReactionOperator.range.id), basis, None).as_range_array().to_numpy().ravel()

        # rho[e] = 1
        #Wenn ich das rho ändere, muss ich garnicht den Operator mit einem neuen Rho konstruieren, da sich das rho automatisch mit verändert, wieso?
        # NonlinearReactionOperator_e = NonlinearReactionOperator_e.with_(rho = rho)
        # # Das projezieren dauert zu lange, deswegen muss ich diesen obigen Weg gehen.
        # reduced_NonlinearReactionOperator_e = project(NonlinearReactionOperator_e, basis, basis)
        # reduced_residual_e = reduced_NonlinearReactionOperator_e.apply(u_ROM).to_numpy().ravel()
        C_rp_ub.append(-reduced_residual_e)
        C_rp_lb.append(reduced_residual_e)

        reduced_element_contribution_jacobian_NonlinearReactionOperator = project(element_contribution_jacobian_NonlinearReactionOperator[e], basis, basis)
        reduced_j_d_u_e = project(element_contribution_jacobian_j[e], basis, None).as_range_array()
        reduced_r_lambda_e = (reduced_element_contribution_jacobian_NonlinearReactionOperator.apply_adjoint(lam_ROM) - reduced_j_d_u_e).to_numpy().ravel()
        C_ra_ub.append(-reduced_r_lambda_e)
        C_ra_lb.append(reduced_r_lambda_e)
        #Da r_d_mu linear und affine Zerlegung hat, kann ich es ignorieren!
        reduced_g_lambda_e = element_contribution_j_d_mu[e] #- np.squeeze(np.array([reduced_r_d_mu_1.apply2(lam_ROM, u_ROM, mu = mu_k),
                                                            #                       reduced_r_d_mu_2.apply2(lam_ROM, u_ROM, mu = mu_k)]))
        C_ga_ub.append(-reduced_g_lambda_e)
        C_ga_lb.append(reduced_g_lambda_e)

        reduced_j_e = np.squeeze(element_contribution_j[e].to_numpy().ravel())
        C_q_ub.append(-reduced_j_e)
        C_q_lb.append(reduced_j_e)

        reduced_r_partial_e_1 = reduced_element_contribution_jacobian_NonlinearReactionOperator.apply(d_mu_1_u_ROM).to_numpy().ravel()
        #reduced_r_partial_e_1 = reduced_NonlinearReactionOperator_e.jacobian(u_ROM).apply(d_mu_1_u_ROM).to_numpy().ravel()
        C_rs_ub_1.append(-reduced_r_partial_e_1)
        C_rs_lb_1.append(reduced_r_partial_e_1)

        reduced_r_partial_e_2 = reduced_element_contribution_jacobian_NonlinearReactionOperator.apply(d_mu_2_u_ROM).to_numpy().ravel()
        #reduced_r_partial_e_2 = reduced_NonlinearReactionOperator_e.jacobian(u_ROM).apply(d_mu_2_u_ROM).to_numpy().ravel()
        C_rs_ub_2.append(-reduced_r_partial_e_2)
        C_rs_lb_2.append(reduced_r_partial_e_2)

        #rho[e] = 0

    C_rp_ubT = np.array(C_rp_ub).T.tolist()
    C_rp_lbT = np.array(C_rp_lb).T.tolist()
    C_ra_ubT = np.array(C_ra_ub).T.tolist()
    C_ra_lbT = np.array(C_ra_lb).T.tolist()
    C_ga_ubT = np.array(C_ga_ub).T.tolist()
    C_ga_lbT = np.array(C_ga_lb).T.tolist()
    C_rs_ub_1T = np.array(C_rs_ub_1).T.tolist()
    C_rs_lb_1T = np.array(C_rs_lb_1).T.tolist()
    C_rs_ub_2T = np.array(C_rs_ub_2).T.tolist()
    C_rs_lb_2T = np.array(C_rs_lb_2).T.tolist()

    # summe1 = 0
    # summe2 = np.zeros(grid.size(2))
    # summe3 = np.zeros(2)
    #
    # for e in range(grid.size(0)):
    #     summe1 = summe1 + np.squeeze(element_contribution_j[e].to_numpy())
    #     summe2 = summe2 + element_contribution_jacobian_j[e].as_range_array().to_numpy().ravel()
    #     summe3 = summe3 + element_contribution_j_d_mu[e]
    #
    # tesr1 = j.apply(basis.lincomb(u_ROM.to_numpy()), mu = mu)
    # tesr2 = j.jacobian(basis.lincomb(u_ROM.to_numpy())).as_range_array()
    # tesr3 = j.d_mu(mu, index = 0)
    # tesr4 = j.d_mu(mu, index = 1)
    #
    #
    # ones = np.ones(grid.size(0))
    # test1 = np.array(C_rp_ubT).dot(ones)
    # test11 = np.array(C_rp_lbT).dot(ones)
    # test111 = rom.operators[4].apply(u_ROM)
    #
    # test2 = np.array(C_ra_ubT).dot(ones)
    # test22 = np.array(C_ra_lbT).dot(ones)
    # J = j.jacobian(basis.lincomb(u_ROM.to_numpy()))
    # pop = project(J, basis, None)
    # reduced_j_d_u = pop.as_range_array()
    # test222 = rom.operators[4].jacobian(u_ROM).apply_adjoint(lam_ROM) - reduced_j_d_u
    #
    # test3 = np.array(C_ga_ubT).dot(ones)
    # test33 = np.array(C_ga_lbT).dot(ones)
    # reduced_j_d_mu = np.array([np.squeeze(j.d_mu(mu_k, index=0)), np.squeeze(j.d_mu(mu_k, index=1))])
    # #wertwert = grid.size(0) * - np.squeeze(np.array([reduced_r_d_mu_1.apply2(lam_ROM, u_ROM, mu=mu_k),
    # #                                                            reduced_r_d_mu_2.apply2(lam_ROM, u_ROM, mu=mu_k)]))
    # test333 = reduced_j_d_mu #- np.squeeze(np.array([reduced_r_d_mu_1.apply2(lam_ROM, u_ROM, mu=mu_k),
    #                          #                                    reduced_r_d_mu_2.apply2(lam_ROM, u_ROM, mu=mu_k)]))
    # test4 = np.array(C_q_ub).dot(ones)
    # test44 = np.array(C_q_lb).dot(ones)
    # test444 = j.apply(basis.lincomb(u_ROM.to_numpy()), mu = mu)
    constraints_A = [C_dv_ub] + [C_dv_lb] + C_rp_ubT + C_rp_lbT + C_ra_ubT + C_ra_lbT + C_ga_ubT + C_ga_lbT + [C_q_ub] + [C_q_lb] + C_rs_ub_1T + C_rs_lb_1T + C_rs_ub_2T + C_rs_lb_2T

    return constraints_A

def constraints_b(volume_omega, residual, adjoint_residual, reduced_grad_f_mu, reduced_f_mu, sensitivity_residual_1, sensitivity_residual_2, mu, delta, basis):

    delta_dv_ub = []
    delta_dv_lb = []
    delta_rp_ub = []
    delta_rp_lb = []
    delta_ra_ub = []
    delta_ra_lb = []
    delta_ga_ub = []
    delta_ga_lb = []
    delta_q_ub = []
    delta_q_lb = []
    delta_rs_ub_1 = []
    delta_rs_lb_1 = []
    delta_rs_ub_2 = []
    delta_rs_lb_2 = []

    delta_dv_ub.append(delta[0] - volume_omega)
    delta_dv_lb.append(delta[0] + volume_omega)
    delta_ga_ub.append(delta[3] - reduced_grad_f_mu[0])
    delta_ga_ub.append(delta[3] - reduced_grad_f_mu[1])
    delta_ga_lb.append(delta[3] + reduced_grad_f_mu[0])
    delta_ga_lb.append(delta[3] + reduced_grad_f_mu[1])
    delta_q_ub.append(delta[4] - reduced_f_mu)
    delta_q_lb.append(delta[4] + reduced_f_mu)
    for i in range(len(basis)):
        delta_rp_ub.append(delta[1] - residual[i])
        delta_rp_lb.append(delta[1] + residual[i])
        delta_ra_ub.append(delta[2] - adjoint_residual[i])
        delta_ra_lb.append(delta[2] + adjoint_residual[i])
        delta_rs_ub_1.append(delta[5] - sensitivity_residual_1[i])
        delta_rs_lb_1.append(delta[5] + sensitivity_residual_1[i])
        delta_rs_ub_2.append(delta[5] - sensitivity_residual_2[i])
        delta_rs_lb_2.append(delta[5] + sensitivity_residual_2[i])


    constraints_b = (delta_dv_ub + delta_dv_lb + delta_rp_ub  + delta_rp_lb + delta_ra_ub + delta_ra_lb +
                     delta_ga_ub + delta_ga_lb + delta_q_ub + delta_q_lb + delta_rs_ub_1 + delta_rs_lb_1 + delta_rs_ub_2 + delta_rs_lb_2)

    return constraints_b

set_log_levels({'pymor': 'INFO'})


domain = RectDomain(([-1,-1], [1,1]))
indicator_domain = ExpressionFunction(
    '(-2/3. <= x[0]) * (x[0] <= -1/3.) * (-2/3. <= x[1]) * (x[1] <= -1/3.) * 1. + (-2/3. <= x[0]) * (x[0] <= -1/3.) * (1/3. <= x[1]) * (x[1] <= 2/3.) * 1.', dim_domain = 2)
rest_of_domain = ConstantFunction(1,2) - indicator_domain
l = ExpressionFunction('0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', dim_domain = 2)
parameters = {'diffusion': 2}
thetas = [ExpressionParameterFunctional('1.1 + sin(diffusion[0])*diffusion[1]', parameters,
                                        derivative_expressions = {'diffusion' : ['cos(diffusion[0])*diffusion[1]',
                                                                                 'sin(diffusion[0])']}),
          ExpressionParameterFunctional('1.1 + sin(diffusion[1])', parameters,
                                        derivative_expressions = {'diffusion':['0',
                                                                               'cos(diffusion[1])']}),]

diffusion = LincombFunction([rest_of_domain, indicator_domain], thetas)
theta_J = ExpressionParameterFunctional('1+1/5 * diffusion[0] +1/5* diffusion[1]', parameters,
                                        derivative_expressions = {'diffusion':['1/5', '1/5']})
linear_reaction_coefficient = ConstantFunction(1,2)
nonlinear_reaction_coefficient = ConstantFunction(1,2)
nonlinear_reaction_function = ExpressionFunction('- u[0] * u[0]', dim_domain = 1, variable = 'u')
nonlinear_reaction_function_derivative = ExpressionFunction('- 2 * u[0]', dim_domain = 1, variable = 'u')

problem = stationary_problem.StationaryProblem(domain = domain, rhs = l, diffusion = diffusion, reaction = linear_reaction_coefficient,  nonlinear_reaction_coefficient = nonlinear_reaction_coefficient,
                                               nonlinear_reaction = nonlinear_reaction_function, nonlinear_reaction_derivative = nonlinear_reaction_function_derivative)
diameter = 1/2
fom, data = discretizer(problem, diameter = diameter)
grid, boundary_info = discretize_domain_default(problem.domain, diameter=diameter)
print('Anzahl Element', grid.size(0))
print('Anzahl DoFs', grid.size(2))
r_d_mu_1 = fom.operator.d_mu('diffusion', index = 0)
r_d_mu_2 = fom.operator.d_mu('diffusion', index = 1)
NonlinearReactionOperator_e = element_NonlinearReactionOperator(grid, boundary_info,
                                                                           nonlinear_reaction_coefficient,
                                                                           nonlinear_reaction_function,
                                                                           nonlinear_reaction_function_derivative,
                                                                           rho = None)

mu_d = problem.parameters.parse(np.array([5,5]))
print('Berechne u_d')
u_d = newton(fom.operator, fom.rhs.as_range_array(), mu = problem.parameters.parse(mu_d))[0]
#fom.visualize(u_d)

j = quadratic_functional(grid, u_d, mu_d)
j_e = element_quadratic_functional(grid, u_d, mu_d, rho = None)



data = {'optimum': [], 'optimum_value':[], 'optimum_grad': [], 'optimum_hess':[], 'num_evals': 0,'num_evals_successful' : 0,
        'evaluation_points': [], 'delta': [], 'gradient_norm': [], 'solving_FOM_pde':  0, 'solving_ROM_pde': 0,
        'solving_hyperreduced_pde': 0, 'length_reduced_basis': [], 'time': 0, 'compute_linprog': 0}

initial_guess = problem.parameters.parse(np.array([4,4]))
delta_0 = 1
delta_max = 1
shrinking_factor = 0.5
error = 10**(-8)
h = 10**(-6)
eta_1=0.1
eta_2=0.9
gamma_1=0.5
gamma_2=0.5
maxiter = 30
delta_EQP = [1e-4,0.5,0.5,0.5,1e-4,1e-4]
kappa_hat = 1
kappa_1 = 10**4
kappa_2 = 10**4
kappa_3 = 10**4
c = np.ones(grid.size(0))
set = fom.solution_space.empty()
U_k = fom.solution_space.empty()
V_k = fom.solution_space.empty()


print('Starte mit dem Algorithmus')
mu_k = initial_guess
data['evaluation_points'].append(mu_k.to_numpy())
delta_k = delta_0
data['delta'].append(delta_k)

#Berechne u_FOM und lam_FOM
data['time'] = perf_counter()
print('löse FOM PDE mit mu_k = ', mu_k)
u_FOM = newton(fom.operator, fom.rhs.as_range_array(), mu = problem.parameters.parse(mu_k))[0]

data['solving_FOM_pde'] += 1
j_d_u = j.jacobian(u_FOM, mu = mu_k).as_range_array()
lam_FOM = fom.operator.jacobian(u_FOM, mu = problem.parameters.parse(mu_k)).apply_inverse_adjoint(j_d_u)

set.append(u_FOM)
set.append(lam_FOM)


# Erstelle die reduzierte Basis
try:
    basis = gram_schmidt(set, product=fom.h1_semi_product, atol=0.0, rtol=0.0)
except:
    print('Extension failed')
data['length_reduced_basis'].append(len(basis))
print('Länge der Basis',len(basis))

rom = LincombOperator([project(op, basis, basis) for op in fom.operator.operators], [1, problem.diffusion.coefficients[0], problem.diffusion.coefficients[1], 1, 1])
rom_rhs = project(fom.rhs, basis, None)
reduced_r_d_mu_1 = rom.d_mu('diffusion', index = 0)
reduced_r_d_mu_2 = rom.d_mu('diffusion', index = 1)
#Mit project klappt das nicht.. deswegen mache ich das selber!
#reduced_j = project(j, None, basis)


print('Löse ROM mit mu_k = ', mu_k)
u_ROM = newton(rom, rom_rhs.as_range_array(), mu = problem.parameters.parse(mu_k))[0]
data['solving_ROM_pde'] += 1


print('berechne reduzierte adjungierte Lösung')
#Hier kann man die vorgeschriebene Porjektion  für operator.jacobian nicht direkt verwenden, da wir
#hier das Minimierungsfunktional betrachten???!!!
J = j.jacobian(basis.lincomb(u_ROM.to_numpy()))
pop = project(J, basis, None)
reduced_j_d_u = pop.as_range_array()
lam_ROM = rom.jacobian(u_ROM, mu = mu_k).apply_inverse_adjoint(reduced_j_d_u)

#Hier bin ich ein bisschen faul und nutze direkt aus, dass die Ableitung von j nach mu nicht mehr von
#u abhängt -> projektion nicht erforderlich!
reduced_j_d_mu = np.array([np.squeeze(j.d_mu(mu_k, index = 0)), np.squeeze(j.d_mu(mu_k, index = 1))])
reduced_grad_f_mu_k = reduced_j_d_mu - np.squeeze(np.array([reduced_r_d_mu_1.apply2(lam_ROM, u_ROM, mu = mu_k),
                                                    reduced_r_d_mu_2.apply2(lam_ROM, u_ROM, mu = mu_k)]))

#sensitivity residual berechnen!
d_mu_1_u_ROM = rom.jacobian(u_ROM, mu = mu_k).apply_inverse(reduced_r_d_mu_1.apply(u_ROM, mu = mu_k))
d_mu_2_u_ROM = rom.jacobian(u_ROM, mu = mu_k).apply_inverse(reduced_r_d_mu_2.apply(u_ROM, mu = mu_k))

#berechne rho_k iterativ, sodass die Toleranzen erfüllt sind
hyperreduced_gradient_f_norm = np.linalg.norm(reduced_grad_f_mu_k)
delta_EQP[1] = kappa_hat/(3*kappa_1) * min(hyperreduced_gradient_f_norm,delta_k)
delta_EQP[2] = kappa_hat/(3*kappa_2) * min(hyperreduced_gradient_f_norm,delta_k)
delta_EQP[3] = kappa_hat/(3*kappa_3) * min(hyperreduced_gradient_f_norm,delta_k)


iteration_rho_k = 0
while True:
    if iteration_rho_k == 0:
        print('berechne constraints A')
        volume_omega = sum(grid.volumes(0).tolist())
        residual = rom.operators[4].apply(u_ROM).to_numpy().ravel()
        J = j.jacobian(basis.lincomb(u_ROM.to_numpy()))
        pop = project(J, basis, None)
        reduced_j_d_u = pop.as_range_array()
        adjoint_residual = rom.operators[4].jacobian(u_ROM).apply_adjoint(lam_ROM).to_numpy().ravel() - reduced_j_d_u.to_numpy().ravel()
        sensitivity_residual_1 = rom.operators[4].jacobian(u_ROM).apply(d_mu_1_u_ROM).to_numpy().ravel()
        sensitivity_residual_2 = rom.operators[4].jacobian(u_ROM).apply(d_mu_2_u_ROM).to_numpy().ravel()
        #reduced_grad_f_mu_k habe ich bereits oben ausgerechnet! Vom reduced_grad_f_mu_k brauchen wir nur den Teil von j_d_mu, da der andere Teil linear und eine Zerlegung besitzt
        reduced_grad_f_mu_k_test = np.array([np.squeeze(j.d_mu(mu_k, index = 0)), np.squeeze(j.d_mu(mu_k, index = 1))])
        reduced_f_mu_k = j.apply(basis.lincomb(u_ROM.to_numpy()), mu = mu_k)
        A = constraints_A(grid, u_ROM, lam_ROM, d_mu_1_u_ROM, d_mu_2_u_ROM, NonlinearReactionOperator_e, mu_k, basis)
    print('berechne constraints b')
    b = constraints_b(volume_omega, residual, adjoint_residual, reduced_grad_f_mu_k_test, reduced_f_mu_k, sensitivity_residual_1, sensitivity_residual_2, mu_k, delta_EQP, basis)
    print('berechne lineares Problem')
    res = linprog(c, A_ub=A, b_ub=b, bounds= (0, None))
    data['compute_linprog'] += 1
    print('wir sind gerade bei mu_k = ', mu_k)
    print(res.message)
    ones = np.ones(grid.size(0))
    test1 = (np.array(A).dot(ones) <= np.array(b)).all()
    rho_k = res.x
    if res.x is None:
        rho_k = np.ones(grid.size(0))
    test21321 = (np.array(A).dot(rho_k) <= np.array(b)).all()
    test21299 = np.array(A).dot(rho_k) - np.array(b)
    NonZeroIndices = np.where(rho_k != 0)[0]
    print(len(NonZeroIndices), 'Einträge sind nicht Null von ', grid.size(0), 'vielen Einträgen')
    #NonlinearReactioOperator für u_ROM!
    NonlinearReactionOperator_e = NonlinearReactionOperator_e.with_(rho = rho_k)
    j_e = j_e.with_(rho = rho_k)
    hyperreduced_NonlinearReactionOperator = project(NonlinearReactionOperator_e, basis, basis)
    hyperreduced_lhs = LincombOperator([rom.operators[0], rom.operators[1], rom.operators[2], rom.operators[3], hyperreduced_NonlinearReactionOperator], [1, problem.diffusion.coefficients[0], problem.diffusion.coefficients[1], 1, 1])

    print('löse hyperreduced PDE mit mu_k =', mu_k)
    u_HROM = newton(hyperreduced_lhs, rom_rhs.as_range_array(), mu = mu_k)[0]
    data['solving_hyperreduced_pde'] += 1
    J = j_e.jacobian(basis.lincomb(u_HROM.to_numpy()))
    pop = project(J, basis, None)
    hyperreduced_j_d_u = pop.as_range_array()
    lam_HROM = hyperreduced_lhs.jacobian(u_HROM, mu = mu_k).apply_inverse_adjoint(hyperreduced_j_d_u)

    hyperreduced_r_d_mu_1_op = hyperreduced_lhs.d_mu('diffusion', index = 0)
    hyperreduced_r_d_mu_2_op = hyperreduced_lhs.d_mu('diffusion', index = 1)

    # Hier bin ich ein bisschen faul und nutze direkt aus, dass die Ableitung von j nach mu nicht mehr von
    # u abhängt -> projektion nicht erforderlich!
    hyperreduced_j_d_mu = np.array([np.squeeze(j_e.d_mu(mu_k, index=0)), np.squeeze(j_e.d_mu(mu_k, index=1))])
    #Obwohl dort hyperreduced_r_d_mu_i steht, nehme ich den reduced_r_d_mu_i!!!
    hyperreduced_grad_f_mu_k = hyperreduced_j_d_mu - np.squeeze(np.array([hyperreduced_r_d_mu_1_op.apply2(lam_HROM, u_HROM, mu = mu_k),
                                                                          hyperreduced_r_d_mu_2_op.apply2(lam_HROM, u_HROM, mu = mu_k)]))

    if (delta_EQP[1] <= kappa_hat/(3*kappa_1) * min(np.linalg.norm(hyperreduced_grad_f_mu_k), delta_k) and
        delta_EQP[2] <= kappa_hat/(3*kappa_2) * min(np.linalg.norm(hyperreduced_grad_f_mu_k), delta_k) and
        delta_EQP[3] <= kappa_hat/(3*kappa_3) * min(np.linalg.norm(hyperreduced_grad_f_mu_k), delta_k)) or all(rho_k == 1):
        #data['gradient_norm'].append(hyperreduced_gradient_f_norm)
        #data['gradient_norm'].append(np.linalg.norm(hyperreduced_grad_f_mu_k))
        break
    delta_EQP[1] = delta_EQP[1] * shrinking_factor * 10**(-1)
    delta_EQP[2] = delta_EQP[2] * shrinking_factor * 10**(-1)
    delta_EQP[3] = delta_EQP[3] * shrinking_factor * 10**(-1)
    iteration_rho_k += 1



n = np.shape(mu_k.to_numpy())[0]
hyperreduced_hess_f_mu_k = np.zeros((n, n))
e_j = np.zeros(n)
for i in range(n):
    e_j[i] = 1
    mu_hess = problem.parameters.parse(mu_k.to_numpy() + h * e_j)
    # berechne die Lösung für initial_guess + h * e_j
    print('löse hyperreduced PDE mit mu_k =', mu_hess)
    u_HROM_hess = newton(hyperreduced_lhs, rom_rhs.as_range_array(), mu = mu_hess)[0]
    data['solving_hyperreduced_pde'] += 1
    J = j_e.jacobian(basis.lincomb(u_HROM_hess.to_numpy()))
    pop = project(J, basis, None)
    hyperreduced_j_d_u_hess = pop.as_range_array()
    lam_HROM_hess = hyperreduced_lhs.jacobian(u_HROM_hess, mu = mu_hess).apply_inverse_adjoint(hyperreduced_j_d_u_hess)


    #berechne hyperreduzierten Gradienten für mu_k = initial_guess
    #hyperreduced_j_d_mu_hess = sum([(grid.volumes(0)[e]/4) * (mu_k + h * e_j - mu_d) for e in NonZeroIndices]) #4 ist das Volumen des Gebietes
    hyperreduced_j_d_mu_hess = np.array([np.squeeze(j_e.d_mu(mu_hess, index=0)), np.squeeze(j_e.d_mu(mu_hess, index=1))])
    hyperreduced_grad_f_hess = hyperreduced_j_d_mu_hess - np.squeeze(np.array([hyperreduced_r_d_mu_1_op.apply2(lam_HROM_hess, u_HROM_hess, mu = mu_hess),
                                                                          hyperreduced_r_d_mu_2_op.apply2(lam_HROM_hess, u_HROM_hess, mu = mu_hess)]))

    hyperreduced_hess_f_mu_k[i, :] = (hyperreduced_grad_f_hess - hyperreduced_grad_f_mu_k) / h
    e_j[i] = 0

f_mu_k = j.apply(u_FOM, mu = mu_k)
j_d_mu = np.array([np.squeeze(j.d_mu(mu_k, index=0)), np.squeeze(j.d_mu(mu_k, index=1))])
grad_f_mu_k = j_d_mu - np.squeeze(np.array([r_d_mu_1.apply2(lam_FOM, u_FOM, mu = mu_k),
                                            r_d_mu_2.apply2(lam_FOM, u_FOM, mu = mu_k)]))
norm = np.linalg.norm(grad_f_mu_k)
data['gradient_norm'].append(norm)
iteration = 0
U_k.append(u_FOM)
V_k.append(lam_FOM)



while True:
    if iteration >= maxiter:
        print('Maximale Anzahle an Iterationen erreicht')
        data['time'] = perf_counter() - data['time']
        data['optimum'] = mu_k
        data['optimum_value'] = f_mu_k_new
        data['optimum_grad'] = grad_f_mu_k
        n = np.shape(mu_k.to_numpy())[0]
        hess_f_mu_k = np.zeros((n, n))
        e_j = np.zeros(n)
        for i in range(n):
            e_j[i] = 1
            mu_hess = problem.parameters.parse(mu_k.to_numpy() + h * e_j)
            u_hess = newton(fom.operator, fom.rhs.as_range_array(), mu=mu_hess)[0]
            data['solving_FOM_pde'] += 1
            j_d_u_hess = j.jacobian(u_FOM).as_range_array()
            lam_hess = fom.operator.jacobian(u_hess, mu=mu_hess).apply_inverse_adjoint(j_d_u_hess)
            j_d_mu_hess = np.array([np.squeeze(j.d_mu(mu_hess, index=0)), np.squeeze(j.d_mu(mu_hess, index=1))])
            grad_f_hess = j_d_mu_hess - np.squeeze(np.array([r_d_mu_1.apply2(lam_hess, u_hess, mu=mu_hess),
                          r_d_mu_2.apply2(lam_hess, u_hess, mu=mu_hess)]))
            hess_f_mu_k[i, :] = (grad_f_hess - grad_f_mu_k) / h
            e_j[i] = 0
        data['optimum_hess'] = hess_f_mu_k
        print(data['evaluation_points'])
        break

    if norm < error:
        print('Norm des Gradienten ist kleiner als der Fehler')
        data['time'] = perf_counter() - data['time']
        data['optimum'] = mu_k
        data['optimum_value'] = f_mu_k_new
        data['optimum_grad'] = grad_f_mu_k
        n = np.shape(mu_k.to_numpy())[0]
        hess_f_mu_k = np.zeros((n, n))
        e_j = np.zeros(n)
        for i in range(n):
            e_j[i] = 1
            mu_hess = problem.parameters.parse(mu_k.to_numpy() + h * e_j)
            u_hess = newton(fom.operator, fom.rhs.as_range_array(), mu=mu_hess)[0]
            data['solving_FOM_pde'] += 1
            j_d_u_hess = j.jacobian(u_FOM).as_range_array()
            lam_hess = fom.operator.jacobian(u_hess, mu=mu_hess).apply_inverse_adjoint(j_d_u_hess)
            j_d_mu_hess = np.array([np.squeeze(j.d_mu(mu_hess, index=0)), np.squeeze(j.d_mu(mu_hess, index=1))])
            grad_f_hess = j_d_mu_hess - np.squeeze(np.array([r_d_mu_1.apply2(lam_hess, u_hess, mu=mu_hess),
                                                             r_d_mu_2.apply2(lam_hess, u_hess, mu=mu_hess)]))
            hess_f_mu_k[i, :] = (grad_f_hess - grad_f_mu_k) / h
            e_j[i] = 0
        data['optimum_hess'] = hess_f_mu_k
        print(f' mu_min: {data["optimum"]}')
        print(f' f(mu_min): {data["optimum_value"]}')
        print(f' grad(mu_min): {data["optimum_grad"]}')
        print(f' hess(mu_min): {data["optimum_hess"]}')
        print(f' num iterations: {data["num_evals"]}')
        print(f' num iterations only successful: {data["num_evals_successful"]}')
        print(f' num solving FOM_pde: {data["solving_FOM_pde"]}')
        print(f' num solving ROM_pde: {data["solving_ROM_pde"]}')
        print(f' num solving Hyperreduced_pde: {data["solving_hyperreduced_pde"]}')
        print(f' num compute linprog: {data["compute_linprog"]}')
        print(f' length basis: {data["length_reduced_basis"]}')
        print(f' time: {data["time"]}')
        print(data['evaluation_points'])
        print(data['gradient_norm'])
        break

    iteration += 1
    # löse das Unterproblem
    print('löse das Unterproblem')
    s_k = trust_region_method.steighaug_toint_truncated_conjugate_gradient_method(g=hyperreduced_grad_f_mu_k,
                                                                                  H=hyperreduced_hess_f_mu_k,
                                                                                  delta=delta_k, k_max=2)
    mu_k_new = problem.parameters.parse(s_k + mu_k.to_numpy())


    # berechne rho_k, d.h. insbesondere, dass man f(mu_k_plus_1) berechnen muss, also man muss die PDE nochmal an mu_k_plus_1 lösen. Vielleicht kann man die Lösung behalten, wenn der Schritt
    # akzeptiert wird
    def model_function(x, g, H):
        return np.dot(g, x - mu_k.to_numpy()) + 0.5 * np.dot(x - mu_k.to_numpy(), np.dot(H, x - mu_k.to_numpy()))

    print('löse FOM PDE mit mu_k = ', mu_k)
    u_FOM_new = newton(fom.operator, fom.rhs.as_range_array(), mu=mu_k_new)[0]
    data['solving_FOM_pde'] += 1
    f_mu_k_new = j.apply(u_FOM_new, mu_k_new)
    ratio = (f_mu_k - f_mu_k_new) / (-np.dot(hyperreduced_grad_f_mu_k, s_k) - 0.5 * np.dot(s_k, np.dot(hyperreduced_hess_f_mu_k, s_k)))

    if ratio >= eta_1:
        print('der Schritt ist erfolgreich')
        mu_k = mu_k_new
        data['evaluation_points'].append(mu_k)
        data['num_evals_successful'] += 1

        #Berechne neue Basis
        set = fom.solution_space.empty()

        # Berechne die Lösung u des FOM für mu_k
        u_FOM = u_FOM_new
        #u_FOM = newton(lhs, F_op.as_range_array(), mu=problem.parameters.parse(mu_k))[0]

        # Berchne die adjungierte Lösung des FOM lambda für mu_k
        j_d_u = j.jacobian(u_FOM, mu=mu_k).as_range_array()
        lam_FOM = fom.operator.jacobian(u_FOM, mu=mu_k).apply_inverse_adjoint(j_d_u)

        set.append(u_FOM)
        set.append(lam_FOM)
        # phi_k_p, _ = pod(U_k, product=fom.h1_semi_product, modes= len(U_k), rtol = 0.0)
        # phi_k_a, __ = pod(V_k, product=fom.h1_semi_product, modes= len(V_k), rtol = 0.0)
        phi_k_p, _ = pod(U_k, product=fom.h1_semi_product, modes= len(U_k))#, rtol = 0.0)
        phi_k_a, __ = pod(V_k, product=fom.h1_semi_product, modes= len(V_k))#, rtol = 0.0)
        set.append(phi_k_p)
        set.append(phi_k_a)

        # Erstelle die reduzierte Basis
        try:
            #Wenn ich das atol = 0, und rtol = 0 angebe, dann funktioniert der Algorithmus für diameter = 2 nicht, da die Basis plötzlich Werte von unendlich besitzt.
            basis = gram_schmidt(set, product=fom.h1_semi_product)#, atol=0.0, rtol=0.0)
        except:
            print('Extension failed')
        data['length_reduced_basis'].append(len(basis))

        rom = LincombOperator([project(op, basis, basis) for op in fom.operator.operators],
                              [1, problem.diffusion.coefficients[0], problem.diffusion.coefficients[1], 1, 1])
        rom_rhs = project(fom.rhs, basis, None)
        reduced_r_d_mu_1 = rom.d_mu('diffusion', index=0)
        reduced_r_d_mu_2 = rom.d_mu('diffusion', index=1)
        #[array([4, 4]), array([4.56329986, 4.82625254]), array([4.99884835, 4.99860275]), array([5.00000002, 4.99999999]), array([5., 5.])]
        print('berechne ROM mit mu_k = ', mu_k)
        print(data['length_reduced_basis'])
        print(data['evaluation_points'])
        print(rom)
        print(rom_rhs.as_range_array())
        u_ROM = newton(rom, rom_rhs.as_range_array(), mu=mu_k)[0]
        data['solving_ROM_pde'] += 1

        print('berechne reduzierte adjungierte Lösung')
        J = j.jacobian(basis.lincomb(u_ROM.to_numpy()))
        pop = project(J, basis, None)
        reduced_j_d_u = pop.as_range_array()
        lam_ROM = rom.jacobian(u_ROM, mu=mu_k).apply_inverse_adjoint(reduced_j_d_u)

        # sensitivity residual berechnen!
        d_mu_1_u_ROM = rom.jacobian(u_ROM, mu=mu_k).apply_inverse(reduced_r_d_mu_1.apply(u_ROM, mu=mu_k))
        d_mu_2_u_ROM = rom.jacobian(u_ROM, mu=mu_k).apply_inverse(reduced_r_d_mu_2.apply(u_ROM, mu=mu_k))

        print('while Schleife lineares Problem')
        hyperreduced_gradient_f_norm = np.linalg.norm(hyperreduced_grad_f_mu_k) #data['gradient_norm'][-1]
        delta_EQP[1] = kappa_hat / (3 * kappa_1) * min(hyperreduced_gradient_f_norm, delta_k)
        delta_EQP[2] = kappa_hat / (3 * kappa_2) * min(hyperreduced_gradient_f_norm, delta_k)
        delta_EQP[3] = kappa_hat / (3 * kappa_3) * min(hyperreduced_gradient_f_norm, delta_k)

        iteration_rho_k = 0

        while True:
            if iteration_rho_k == 0:
                print('berechne constraints A')
                volume_omega = sum(grid.volumes(0).tolist())
                residual = rom.operators[4].apply(u_ROM).to_numpy().ravel()
                J = j.jacobian(basis.lincomb(u_ROM.to_numpy()))
                pop = project(J, basis, None)
                reduced_j_d_u = pop.as_range_array()
                adjoint_residual = rom.operators[4].jacobian(u_ROM).apply_adjoint(lam_ROM).to_numpy().ravel() - reduced_j_d_u.to_numpy().ravel()
                sensitivity_residual_1 = rom.operators[4].jacobian(u_ROM).apply(d_mu_1_u_ROM).to_numpy().ravel()
                sensitivity_residual_2 = rom.operators[4].jacobian(u_ROM).apply(d_mu_2_u_ROM).to_numpy().ravel()

                reduced_j_d_mu = np.array([np.squeeze(j.d_mu(mu_k, index=0)), np.squeeze(j.d_mu(mu_k, index=1))])
                # reduced_grad_f_mu_k = reduced_j_d_mu - np.squeeze(np.array([reduced_r_d_mu_1.apply2(lam_ROM, u_ROM, mu=mu_k),
                #                                                             reduced_r_d_mu_2.apply2(lam_ROM, u_ROM, mu=mu_k)]))
                reduced_grad_f_mu_k_test = np.array([np.squeeze(j.d_mu(mu_k, index=0)), np.squeeze(j.d_mu(mu_k, index=1))])
                reduced_f_mu_k = j.apply(basis.lincomb(u_ROM.to_numpy()), mu=mu_k)
                A = constraints_A(grid, u_ROM, lam_ROM, d_mu_1_u_ROM, d_mu_2_u_ROM, NonlinearReactionOperator_e, mu_k,
                                  basis)
            print('berechne constraints b')
            b = constraints_b(volume_omega, residual, adjoint_residual, reduced_grad_f_mu_k_test, reduced_f_mu_k, sensitivity_residual_1, sensitivity_residual_2,
                              mu_k, delta_EQP, basis)
            print('berechne lineares Problem')
            res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
            data['compute_linprog'] += 1
            print('wir sind gerade bei mu_k = ', mu_k)
            print(res.message)
            rho_k = res.x
            test12 = (np.array(A).dot(ones) <= np.array(b)).all()
            if res.x is None:
                rho_k = np.ones(grid.size(0))
            test213 = (np.array(A).dot(rho_k) <= np.array(b)).all()
            test129991 = np.array(A).dot(rho_k) - np.array(b)
            NonZeroIndices = np.where(rho_k != 0)[0]
            print(len(NonZeroIndices), 'Einträge sind nicht Null von ', grid.size(0), 'vielen Einträgen')
            # NonlinearReactioOperator für u_ROM!
            NonlinearReactionOperator_e = NonlinearReactionOperator_e.with_(rho=rho_k)
            j_e = j_e.with_(rho = rho_k)
            hyperreduced_NonlinearReactionOperator = project(NonlinearReactionOperator_e, basis, basis)
            hyperreduced_lhs = LincombOperator([rom.operators[0], rom.operators[1], rom.operators[2], rom.operators[3],
                                                hyperreduced_NonlinearReactionOperator],
                                               [1, problem.diffusion.coefficients[0], problem.diffusion.coefficients[1],
                                                1, 1])

            print('löse hyperreduced PDE mit mu_k =', mu_k)
            u_HROM = newton(hyperreduced_lhs, rom_rhs.as_range_array(), mu=mu_k)[0]
            data['solving_hyperreduced_pde'] += 1
            J = j_e.jacobian(basis.lincomb(u_HROM.to_numpy()))
            pop = project(J, basis, None)
            hyperreduced_j_d_u = pop.as_range_array()
            lam_HROM = hyperreduced_lhs.jacobian(u_HROM, mu=mu_k).apply_inverse_adjoint(hyperreduced_j_d_u)

            hyperreduced_r_d_mu_1_op = hyperreduced_lhs.d_mu('diffusion', index=0)
            hyperreduced_r_d_mu_2_op = hyperreduced_lhs.d_mu('diffusion', index=1)

            # Hier bin ich ein bisschen faul und nutze direkt aus, dass die Ableitung von j nach mu nicht mehr von
            # u abhängt -> projektion nicht erforderlich!
            hyperreduced_j_d_mu = np.array([np.squeeze(j_e.d_mu(mu_k, index=0)), np.squeeze(j_e.d_mu(mu_k, index=1))])
            hyperreduced_grad_f_mu_k = hyperreduced_j_d_mu - np.squeeze(
                np.array([hyperreduced_r_d_mu_1_op.apply2(lam_HROM, u_HROM, mu=mu_k),
                          hyperreduced_r_d_mu_2_op.apply2(lam_HROM, u_HROM, mu=mu_k)]))

            if (delta_EQP[1] <= kappa_hat / (3 * kappa_1) * min(np.linalg.norm(hyperreduced_grad_f_mu_k), delta_k) and
                delta_EQP[2] <= kappa_hat / (3 * kappa_2) * min(np.linalg.norm(hyperreduced_grad_f_mu_k), delta_k) and
                delta_EQP[3] <= kappa_hat / (3 * kappa_3) * min(np.linalg.norm(hyperreduced_grad_f_mu_k),
                                                                delta_k)) or all(rho_k == 1):
                # data['gradient_norm'].append(hyperreduced_gradient_f_norm)
                # data['gradient_norm'].append(np.linalg.norm(hyperreduced_grad_f_mu_k))
                break
            delta_EQP[1] = delta_EQP[1] * shrinking_factor * 10 ** (-1)
            delta_EQP[2] = delta_EQP[2] * shrinking_factor * 10 ** (-1)
            delta_EQP[3] = delta_EQP[3] * shrinking_factor * 10 ** (-1)
            iteration_rho_k += 1

        n = np.shape(mu_k.to_numpy())[0]
        hyperreduced_hess_f_mu_k = np.zeros((n, n))
        e_j = np.zeros(n)
        for i in range(n):
            e_j[i] = 1
            mu_hess = problem.parameters.parse(mu_k.to_numpy() + h * e_j)
            # berechne die Lösung für initial_guess + h * e_j
            print('löse hyperreduced PDE mit mu_k =', mu_hess)
            u_HROM_hess = newton(hyperreduced_lhs, rom_rhs.as_range_array(), mu=mu_hess)[0]
            data['solving_hyperreduced_pde'] += 1
            J = j_e.jacobian(basis.lincomb(u_HROM_hess.to_numpy()))
            pop = project(J, basis, None)
            hyperreduced_j_d_u_hess = pop.as_range_array()
            lam_HROM_hess = hyperreduced_lhs.jacobian(u_HROM_hess, mu=mu_hess).apply_inverse_adjoint(
                hyperreduced_j_d_u_hess)

            # berechne hyperreduzierten Gradienten für mu_k = initial_guess
            # hyperreduced_j_d_mu_hess = sum([(grid.volumes(0)[e]/4) * (mu_k + h * e_j - mu_d) for e in NonZeroIndices]) #4 ist das Volumen des Gebietes
            hyperreduced_j_d_mu_hess = np.array(
                [np.squeeze(j_e.d_mu(mu_hess, index=0)), np.squeeze(j_e.d_mu(mu_hess, index=1))])
            hyperreduced_grad_f_hess = hyperreduced_j_d_mu_hess - np.squeeze(
                np.array([hyperreduced_r_d_mu_1_op.apply2(lam_HROM_hess, u_HROM_hess, mu=mu_hess),
                          hyperreduced_r_d_mu_2_op.apply2(lam_HROM_hess, u_HROM_hess, mu=mu_hess)]))

            hyperreduced_hess_f_mu_k[i, :] = (hyperreduced_grad_f_hess - hyperreduced_grad_f_mu_k) / h
            e_j[i] = 0

        f_mu_k = j.apply(u_FOM, mu=mu_k)
        j_d_mu = np.array([np.squeeze(j.d_mu(mu_k, index=0)), np.squeeze(j.d_mu(mu_k, index=1))])
        grad_f_mu_k = j_d_mu - np.squeeze(np.array([r_d_mu_1.apply2(lam_FOM, u_FOM, mu=mu_k),
                                                    r_d_mu_2.apply2(lam_FOM, u_FOM, mu=mu_k)]))
        norm = np.linalg.norm(grad_f_mu_k)
        data['gradient_norm'].append(norm)
        iteration = 0
        U_k.append(u_FOM)
        V_k.append(lam_FOM)

    if ratio >= eta_2:
        delta_k_plus_1 = delta_k * 1 / shrinking_factor
        if delta_k_plus_1 > delta_max:
            delta_k_plus_1 = delta_max
    elif eta_1 <= ratio < eta_2:
        delta_k_plus_1 = delta_k * shrinking_factor
        if delta_k_plus_1 < gamma_2 * delta_k:
            delta_k_plus_1 = gamma_2 * delta_k
    else:
        delta_k_plus_1 = delta_k * shrinking_factor
        if delta_k_plus_1 < gamma_1 * delta_k:
            delta_k_plus_1 = gamma_1 * delta_k
        if delta_k_plus_1 > gamma_2 * delta_k:
            delta_k_plus_1 = gamma_2 * delta_k
    delta_k = delta_k_plus_1
    data['num_evals'] += 1
    data['delta'].append(delta_k)



