import numpy as np
from typer import Option, run

from pymor.algorithms.phdmd import phdmd
from pymor.models.iosys import PHLTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.ph.ph_irka import PHIRKAReductor
from pymordemos.phlti import msd


def main(
        fom_order: int = Option(50, help='Order of the Mass-Spring-Damper system.'),
        rom_order: int = Option(6, help='Order of the reduced Mass-Spring-Damper system.'),
        final_training_time: float = Option(4., help='Final simulation time for generating the training data.'),
        final_validation_time: float = Option(10., help='Final simulation time for generating the validation data.'),
        dt: float = Option(4e-2, help='Time difference used for simulation.')
):
    """Example script demonstrating the pH DMD algorithm.

    A Mass-Spring-Damper system is used as both a full order model as well as a
    reduced order model. Both of these are then inferred using pH DMD.
    """
    J, R, G, P, S, N, E = msd(n=fom_order)

    fom = PHLTIModel.from_matrices(J, R, G, P, S, N, E)
    phirka = PHIRKAReductor(fom)
    rom = phirka.reduce(rom_order)
    _, _, _, _, _, _, rom_E, _ = rom.to_matrices()

    training_time_stamps = np.arange(0., final_training_time + dt, dt)
    validation_time_stamps = np.arange(0., final_validation_time + dt, dt)

    fom_state_space = fom.solution_space
    rom_state_space = rom.solution_space
    io_space = fom.G.source

    fom_initial = np.zeros(fom_state_space.dim)
    rom_initial = np.zeros(rom_state_space.dim)

    fom_X, U, fom_Y, _ = _implicit_midpoint(fom, excitation_control, fom_initial, training_time_stamps)
    rom_X, _, rom_Y, _ = _implicit_midpoint(rom, excitation_control, rom_initial, training_time_stamps)

    fom_X = fom_state_space.from_numpy(fom_X.T)
    rom_X = rom_state_space.from_numpy(rom_X.T)
    fom_Y = io_space.from_numpy(fom_Y.T)
    rom_Y = io_space.from_numpy(rom_Y.T)
    U = io_space.from_numpy(U.T)

    inf_fom, fom_data = phdmd(fom_X, fom_Y, U, H=E, dt=dt, rtol=1e-8)
    inf_rom, rom_data = phdmd(rom_X, rom_Y, U, H=rom_E, dt=dt)

    _, _, fom_Y, _ = _implicit_midpoint(fom, validation_control(final_validation_time),
                                        fom_initial, validation_time_stamps)
    _, _, inf_Y, _ = _implicit_midpoint(inf_fom, validation_control(final_validation_time),
                                        fom_initial, validation_time_stamps)
    _, _, rom_Y, _ = _implicit_midpoint(rom, validation_control(final_validation_time),
                                        rom_initial, validation_time_stamps)
    _, _, inf_rom_Y, _ = _implicit_midpoint(inf_rom, validation_control(final_validation_time),
                                            rom_initial, validation_time_stamps)

    rel_rom_error = np.abs(fom_Y[0] - rom_Y[0]) / np.linalg.norm(fom_Y[0])
    rel_fom_train_error = fom_data['rel_errs'][-1]
    rel_rom_train_error = rom_data['rel_errs'][-1]
    rel_fom_inf_error = np.abs(fom_Y[0] - inf_Y[0]) / np.linalg.norm(fom_Y[0])
    rel_rom_inf_error = np.abs(rom_Y[0] - inf_rom_Y[0]) / np.linalg.norm(rom_Y[0])

    print(f'Maximum error of ROM solution w.r.t. FOM solution:                    {np.max(rel_rom_error):.3e}')
    print(f'Relative error of training data w.r.t. FOM solution:                  {rel_fom_train_error:.3e}')
    print(f'Relative error of training data w.r.t. ROM solution:                  {rel_rom_train_error:.3e}')
    print(f'Maximum error of full order validation output w.r.t. FOM solution:    {np.max(rel_fom_inf_error):.3e}')
    print(f'Maximum error of reduced order validation output w.r.t. ROM solution: {np.max(rel_rom_inf_error):.3e}')


def _implicit_midpoint(model, control, initial_condition, time_stamps, return_derivatives=False):
    assert isinstance(model, PHLTIModel)
    assert isinstance(model.Q, IdentityOperator), 'PHLTIModel has to be in generalized form!'

    control_snapshots = control(time_stamps)
    if control_snapshots.ndim < 2:
        control_snapshots = control_snapshots[np.newaxis, :]

    dt = time_stamps[1] - time_stamps[0]

    J, R, G, P, S, N, E, _ = model.to_matrices()
    A = J - R
    B = G - P
    C = (G + P).T
    D = S - N

    M = E - (dt / 2.) * A
    AA = E + (dt / 2.) * A

    state_snapshots = np.zeros((model.order, len(time_stamps)))
    state_snapshots[:, 0] = initial_condition

    for i in range(len(time_stamps) - 1):
        control_midpoint = 1 / 2 * (control_snapshots[:, i] + control_snapshots[:, i + 1])
        state_snapshots[:, i + 1] = np.linalg.solve(M, AA @ state_snapshots[:, i] + dt * B @ control_midpoint)

    output_snapshots = C @ state_snapshots + D @ control_snapshots

    if return_derivatives:
        derivative_snapshots = np.linalg.solve(E, A @ state_snapshots + B @ control_snapshots)
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
