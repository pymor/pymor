import matplotlib.pyplot as plt
import numpy as np
from pymor.algorithms.dmd import dmd
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.discretizers.builtin.cg import discretize_instationary_cg
from pymor.discretizers.builtin.fv import discretize_instationary_fv
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.discretizers.builtin.grids.tria import TriaGrid
from typer import Option, run


def main(
        fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
        grid: int = Option(400, help='Use grid with NIxNI intervals.'),
        nt: int = Option(100, help='Number of time steps.'),
        rect: bool = Option(False, help='Use RectGrid instead of TriaGrid.'),
):
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(),

            diffusion=ConstantFunction(0.01, dim_domain=2),

            reaction=ConstantFunction(0.5, dim_domain=2),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction(
            '(x[0] > 0.3) * (x[0] < 0.7) * (x[1]>0.3) * (x[1] < 0.7) * 10.', dim_domain=2),
    )

    print('Discretize ...')
    discretizer = discretize_instationary_fv if fv else discretize_instationary_cg
    m, data = discretizer(
        analytical_problem=problem,
        grid_type=RectGrid if rect else TriaGrid,
        diameter=np.sqrt(2) / grid if rect else 1. / grid,
        nt=nt
    )
    grid = data['grid']
    print(grid)
    print()

    print('Solve ...')
    U = m.solve()

    print('')

    m.visualize(U, title='Solution of Parabolic Problem')

    # ----- Testing DMD -----
    W1, E1 = dmd(X=U, modes='standard')
    W2, E2 = dmd(X=U, target_rank=5, modes='exact')

    print('Visualize ...')
    m.visualize(W1, title='DMD Modes - standard')
    plt.plot(E1.real, E1.imag, 'b.')
    plt.title('DMD Eigenvalues - standard')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()

    m.visualize(W2, title='5 DMD Modes - exact')
    plt.plot(E2.real, E2.imag, 'b.')
    plt.title('DMD Eigenvalues - exact')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


if __name__ == '__main__':
    run(main)
