import numpy as np

from pymor.operators.constructions import ZeroOperator, LincombOperator, VectorOperator
from pymor.algorithms.projection import project
from pymor.operators.block import BlockOperator

from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.gram_schmidt import gram_schmidt

class CoerciveIPLD3GRBReductor(CoerciveRBReductor):
    def __init__(self, fom):
        self.S = fom.solution_space.empty().num_blocks
        self.fom = fom

        self.local_bases = [fom.solution_space.empty().block(ss).empty()
                            for ss in range(self.S)]

    def add_global_solutions(self, us):
        assert us in self.fom.solution_space
        for I in range(self.S):
            us_block = us.block(I)
            self.local_bases[I].append(us_block)
            # TODO: add offset
            self.local_bases[I] = gram_schmidt(self.local_bases[I])

    def add_local_solutions(self, I, u):
        self.local_bases[I].append(u)
        # TODO: add offset
        self.local_bases[I] = gram_schmidt(self.local_bases[I])

    def basis_length(self):
        return [len(self.local_bases[I]) for I in range(self.S)]

    def reduce(self):
        return self._reduce()

    def project_operators(self):
        projected_ops_blocks = []
        # this is for BlockOperator(LincombOperators)
        assert isinstance(self.fom.operator, BlockOperator)
        assert not self.fom.rhs.parametric

        local_projected_op = np.empty((self.S, self.S), dtype=object)
        for I in range(self.S):
            for J in range(self.S):
                local_basis_I = self.local_bases[I]
                local_basis_J = self.local_bases[J]
                if self.fom.operator.blocks[I][J]:
                    local_projected_op[I][J] = project(self.fom.operator.blocks[I][J],
                                                       local_basis_I, local_basis_J)
        projected_operator = BlockOperator(local_projected_op)

        local_projected_rhs = np.empty(self.S, dtype=object)
        for I in range(self.S):
            # TODO: find an easier way for this this is currently not possible for parametric rhs
            local_basis = self.local_bases[I]
            rhs_int = project(self.fom.rhs.blocks[I, 0], local_basis, None).matrix[:, 0]
            local_projected_rhs[I] = local_projected_op[I][I].range.make_array(rhs_int)
        projected_rhs = VectorOperator(projected_operator.range.make_array(local_projected_rhs))
        # projected_rhs = BlockOperator(local_projected_rhs)

        projected_operators = {
            'operator':          projected_operator,
            'rhs':               projected_rhs,
            'products':          None,
            'output_functional': None
        }
        return projected_operators

    def assemble_error_estimator(self):
        return None

    def reconstruct(self, u_rom):
        u_ = []
        for I in range(self.S):
            basis = self.local_bases[I]
            u_I = u_rom.block(I)
            u_.append(basis.lincomb(u_I.to_numpy()))
        return self.fom.solution_space.make_array(u_)
