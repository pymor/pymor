from pymordemos.phlti import msd
from pymordemos.phdmd import solve_ph_lti
from pymor.models.iosys import PHLTIModel

import numpy as np
from typer import run


def main():
    J, R, G, P, S, N, E = msd()

    fom = PHLTIModel.from_matrices(J, R, G, P, S, N, E)


if __name__ == '__main__':
    run(main)