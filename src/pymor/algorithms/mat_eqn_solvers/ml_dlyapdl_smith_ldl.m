function [RZ, RY, LZ, LY, info] = ml_dlyapdl_smith_ldl(A, B, R, C, Q, E, opts)
%ML_DLYAPDL_SMITH_LDL Discrete-time dual Lyapunov equation solver.
%
% SYNTAX:
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, [], C, [])
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, R, C, Q)
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, R, C, Q, [])
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, R, C, Q, [], opts)
%
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, [], C, [], E)
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, R, C, Q, E)
%   [RZ, RY, LZ, LY, info] = ML_DLYAPDL_SMITH_LDL(A, B, R, C, Q, E, opts)
%
% DESCRIPTION:
%   Computes the solution matrix of the dual standard discrete-time
%   Lyapunov equations
%
%       A*X1*A' - X1 + B*R*B' = 0,                                      (1)
%       A'*X2*A - X2 + C'*Q*C = 0,                                      (2)
%
%   or of the dual generalized Lyapunov equations
%
%       A*X1*A' - E*X1*E' + B*R*B' = 0,                                 (3)
%       A'*X2*A - E'*X2*E + C'*Q*C = 0,                                 (4)
%
%   with X1 = RZ*RY*RZ' and X2 = LZ*LY*LZ' using the Smith iteration. It is
%   assumed that the eigenvalues of A (or s*E - A) lie inside the open
%   unit-circle.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1), (2) or (3), (4)
%   B    - matrix with dimensions n x m in (1) or (3)
%   R    - symmetric matrix with dimensions m x m in (1) or (3),
%          if empty R is assumed to be the identity
%   C    - matrix with dimensions p x n in (2) or (4)
%   Q    - symmetric matrix with dimensions p x p in (2) or (4),
%          if empty Q is assumed to be the identity
%   E    - matrix with dimensions n x n in (3), (4),
%          if empty the standard equations (1), (2) are solved
%   opts - structure, containing the following optional entries:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsTol          | nonnegative scalar, tolerance for the absolute    |
%   |                 | error in the last iteration step                  |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | CompTol         | nonnegative scalar, tolerance for the column and  |
%   |                 | row compression during the iteration              |
%   |                 | (default sqrt(n)*eps)                             |
%   +-----------------+---------------------------------------------------+
%   | Info            | {0, 1}, used to disable/enable display of verbose |
%   |                 | status information during iteration steps         |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | MaxIter         | positive integer, maximum number of iteration     |
%   |                 | steps                                             |
%   |                 | (default 100)                                     |
%   +-----------------+---------------------------------------------------+
%   | RelTol          | nonnegative scalar, tolerance for the relative    |
%   |                 | error in the last iteration step                  |
%   |                 | (default 1.0e+01*n*eps)                           |
%   +-----------------+---------------------------------------------------+
%
% OUTPUTS:
%   RZ   - full-rank solution factor of (1) or (3), s.t. X1 = RZ*RY*RZ'
%   RY   - full-rank solution factor of (1) or (3), s.t. X1 = RZ*RY*RZ'
%   LZ   - full-rank solution factor of (2) or (4), s.t. X2 = LZ*LY*LZ'
%   LY   - full-rank solution factor of (2) or (4), s.t. X2 = LZ*LY*LZ'
%   info - structure, containing the following information:
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsErr          | vector, containing the absolute error of the      |
%   |                 | iteration matrix in each iteration step           |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the relative error of the      |
%   |                 | iteration matrix in each iteration step           |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%   V. Simoncini, Computational methods for linear matrix equations, SIAM
%   Rev. 38 (3) (2016) 377--441. https://doi.org/10.1137/130912839
%
% See also ml_dlyap_smith, ml_dlyap_smith_ldl.

%
% This file is part of the MORLAB toolbox
% (https://www.mpi-magdeburg.mpg.de/projects/morlab).
% Copyright (C) 2006-2023 Peter Benner, Jens Saak, and Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause License (see COPYING)
%


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

narginchk(5, 7);

if (nargin < 7) || isempty(opts)
    opts = struct();
end

% Check input matrices.
n = size(A, 1);
m = size(B, 2);
p = size(C, 1);

assert(isequal(size(A), [n n]), ...
    'MORLAB:data', ...
    'The matrix A has to be square!');

assert(size(B, 1) == n, ...
    'MORLAB:data', ...
    'The matrix B must have the same number of rows as A!');

assert(size(C, 2) == n, ...
    'MORLAB:data', ...
    'The matrix C must have the same number of columns as A!');

if issparse(A), A = full(A); end
if issparse(B), B = full(B); end
if issparse(C), C = full(C); end

if not(isempty(R))
    assert(isequal(size(R), [m, m]), ...
        'MORLAB:data', ...
        'The matrix R must be square with the number of columns as B!');

    assert(norm(R - R', 'fro') < 1.0e-14, ...
        'MORLAB:data', ...
        'The matrix R must be symmetric!');

    if issparse(R), R = full(R); end
else
    R = eye(m);
end

if not(isempty(Q))
    assert(isequal(size(Q), [p, p]), ...
        'MORLAB:data', ...
        'The matrix Q must be square with the number of rows as C!');

    assert(norm(Q - Q', 'fro') < 1.0e-14, ...
        'MORLAB:data', ...
        'The matrix Q must be symmetric!');

    if issparse(Q), Q = full(Q); end
else
    Q = eye(p);
end

if (nargin >= 6) && not(isempty(E))
    assert(isequal(size(E), [n n]), ...
        'MORLAB:data', ...
        'The matrix E must have the same dimensions as A!');

    if issparse(E), E = full(E); end

    hasE = 1;
else
    E    = eye(n);
    hasE = 0;
end

% Check and assign optional parameters.
assert(isa(opts, 'struct'), ...
    'MORLAB:data', ...
    'The parameter opts has to be a struct!');

if ml_field_set_to_value(opts, 'AbsTol')
    ml_assert_nonnegscalar(opts.AbsTol, 'opts.AbsTol');
else
    opts.AbsTol = 0;
end

if ml_field_set_to_value(opts, 'CompTol')
    ml_assert_nonnegscalar(opts.CompTol, 'opts.CompTol');
else
    opts.CompTol = sqrt(n) * eps;
end

if ml_field_set_to_value(opts, 'Info')
    ml_assert_boolean(opts.Info, 'opts.Info');
else
    opts.Info = false;
end

if ml_field_set_to_value(opts, 'MaxIter')
    ml_assert_posinteger(opts.MaxIter, 'opts.MaxIter');
else
    opts.MaxIter = 100;
end

if ml_field_set_to_value(opts, 'RelTol')
    ml_assert_nonnegscalar(opts.RelTol, 'opts.RelTol');
else
    opts.RelTol = 1.0e+01 * (n * eps);
end

% Case of empty data.
if isempty(A)
    [RZ, RY, LZ, LY] = deal([]);
    info             = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if hasE
    A  = E \ A;
    RZ = E \ B;
else
    RZ = B;
end
RY = R;
LZ = C;
LY = Q;

niter     = 1;
converged = 0;

[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SQUARED SMITH ITERATION.                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    % Construction of next solution matrix.
    ARZ      = A * RZ;
    ALZ      = LZ * A;
    [RZ, RY] = ml_compress_ldl([RZ, ARZ], blkdiag(RY, RY), ...
        opts.CompTol, 'col');
    [LZ, LY] = ml_compress_ldl([LZ; ALZ], blkdiag(LY, LY), ...
        opts.CompTol, 'row');

    % Update of iteration matrix.
    A = A * A;

    % Information about current iteration step.
    normARZ = norm(ARZ, 'fro');
    normALZ = norm(ALZ, 'fro');
    if normARZ >= normALZ
        abserr(niter) = normARZ;
        relerr(niter) = abserr(niter) / norm(RZ, 'fro');
    else
        abserr(niter) = normALZ;
        relerr(niter) = abserr(niter) / norm(LZ, 'fro');
    end

    if opts.Info
        fprintf(1, ['DLYAPDL_SMITH_LDL step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if hasE
    LZ = (LZ / E)';
else
    LZ = LZ';
end

niter = niter - 1;

% Warning if iteration not converged.
if (niter == opts.MaxIter) && not(converged)
    warning('MORLAB:noConvergence', ...
        ['No convergence in %d iteration steps!\n' ...
        'Abs. tolerance: %e | Abs. error: %e\n' ...
        'Rel. tolerance: %e | Rel. error: %e\n' ...
        'Try to increase the tolerances or number of ' ...
        'iteration steps!'], ...
        niter, opts.AbsTol, abserr(niter), ...
        opts.RelTol, relerr(niter));
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASSIGN INFORMATION.                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assign information about iteration.
info = struct( ...
    'AbsErr'        , abserr(1:niter), ...
    'IterationSteps', niter, ...
    'RelErr'        , relerr(1:niter));
