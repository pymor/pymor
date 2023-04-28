from pymordemos.phlti import msd
from pymor.models.iosys import PHLTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.ph.ph_irka import PHIRKAReductor
from pymor.algorithms.phdmd import phdmd
from pymor.core.logger import getLogger

import numpy as np
from typer import run, Option
from scipy.integrate import solve_ivp
from scipy.linalg import solve

import matplotlib.pyplot as plt


def main(
        fom_order: int = Option(6, help='Order of the Mass-Spring-Damper system.'),
        rom_order: int = Option(20, help='Order of the reduced Mass-Spring-Damper system.'),
        final_training_time: float = Option(4., help='Final simulation time for generating the training data.'),
        final_validation_time: float = Option(10., help='Final simulation time for generating the validation data.'),
        dt: float = Option(1. / 25., help='Time difference used for simulation.'),
        stepper_method: str = Option('RK45', help='Time stepping method employed by SciPy.')
):
    logger = getLogger('pymordemos.phdmd.main')

    J, R, G, P, S, N, E = msd(n=fom_order)

    fom = PHLTIModel.from_matrices(J, R, G, P, S, N, E)
    phirka = PHIRKAReductor(fom)
    # rom = phirka.reduce(rom_order)

    training_time_stamps = np.arange(0., final_training_time + dt, dt)
    validation_time_stamps = np.arange(0., final_validation_time + dt, dt)

    fom_state_space = fom.solution_space
    # rom_state_space = rom.solution_space
    io_space = fom.G.source

    fom_initial = np.zeros(fom_state_space.dim)
    # rom_initial = np.zeros(rom_state_space.dim)

    fom_X, U, fom_Y, fom_derivatives = solve_ph_lti(fom, excitation_control, fom_initial, training_time_stamps, method=stepper_method, return_derivatives=True)
    # rom_X, _, rom_Y, _ = solve_ph_lti(rom, excitation_control, rom_initial, training_time_stamps, method=stepper_method)

    fom_X = fom_state_space.from_numpy(fom_X.T)
    fom_derivatives = fom_state_space.from_numpy(fom_derivatives.T)
    # rom_X = rom_state_space.from_numpy(rom_X.T)
    fom_Y = io_space.from_numpy(fom_Y.T)
    # rom_Y = io_space.from_numpy(rom_Y.T)
    U = io_space.from_numpy(U.T)

    inf_fom, fom_data = phdmd(fom_X, fom_Y, U, dt=dt, maxiter=10000, rtol=.235)
    # inf_rom, rom_data = phdmd(rom_X, rom_Y, U, dt=dt)

    inf_X, U, inf_Y, inf_derivatives = solve_ph_lti(inf_fom, validation_control(final_validation_time), fom_initial, validation_time_stamps, method=stepper_method, return_derivatives=True)
    fom_X, U, fom_Y, fom_derivatives = solve_ph_lti(fom, validation_control(final_validation_time), fom_initial, validation_time_stamps, method=stepper_method, return_derivatives=True)

    plt.figure()
    plt.plot(validation_time_stamps, np.squeeze(inf_Y))
    plt.plot(validation_time_stamps, np.squeeze(fom_Y))
    plt.show()


def solve_ph_lti(model, control, initial_condition, time_stamps, method='RK45', return_derivatives=False):
    assert isinstance(model, PHLTIModel)
    assert isinstance(model.Q, IdentityOperator), 'PHLTIModel has to be in generalized form!'

    time_interval = (time_stamps[0], time_stamps[-1])

    control_snapshots = control(time_stamps)
    if control_snapshots.ndim < 2:
        control_snapshots = control_snapshots[np.newaxis, :]

    J, R, G, P, S, N, E, _ = model.to_matrices()
    A = J - R
    B = G - P
    C = (G + P).T
    D = S - N

    def time_step(t, state, control):
        return solve(E, A @ state + B @ control(t))

    sol = solve_ivp(time_step, time_interval, initial_condition, t_eval=time_stamps, method=method, args=(control,))
    state_snapshots = sol.y
    output_snapshots = C @ state_snapshots + D @ control_snapshots

    if return_derivatives:
        derivative_snapshots = solve(E, A @ state_snapshots + B @ control_snapshots)
    else:
        derivative_snapshots = None
    return state_snapshots, control_snapshots, output_snapshots, derivative_snapshots

def excitation_control(t):
    return np.array([np.exp(-.5 * t) * np.sin(t**2)])

def validation_control(final_time, num_partitions=5, vals=[-1., 1.]):
    partition_len = final_time / num_partitions

    def inner(t):
        proportion = (t % partition_len) / partition_len
        return np.array([vals[0] + proportion * (vals[1] - vals[0])])

    return inner


if __name__ == '__main__':
    run(main)