function [R, L, info] = ml_dlyapdl_smith_fac(A, B, C, E, opts)
%ML_DLYAPDL_SMITH_FAC Discrete-time dual Lyapunov equation solver.
%
% SYNTAX:
%   [R, L, info] = ML_DLYAPDL_SMITH_FAC(A, B, C)
%   [R, L, info] = ML_DLYAPDL_SMITH_FAC(A, B, C, [])
%   [R, L, info] = ML_DLYAPDL_SMITH_FAC(A, B, C, [], opts)
%
%   [R, L, info] = ML_DLYAPDL_SMITH_FAC(A, B, C, E)
%   [R, L, info] = ML_DLYAPDL_SMITH_FAC(A, B, C, E, opts)
%
% DESCRIPTION:
%   Computes the solution matrix of the dual standard discrete-time
%   Lyapunov equations
%
%       A*X*A' - X + B*B' = 0,                                          (1)
%       A'*Y*A - Y + C'*C = 0,                                          (2)
%
%   or of the dual generalized Lyapunov equations
%
%       A*X*A' - E*X*E' + B*B' = 0,                                     (3)
%       A'*Y*A - E'*Y*E + C'*C = 0,                                     (4)
%
%   with X = R*R' and Y = L*L' using the Smith iteration. It is assumed
%   that the eigenvalues of A (or s*E - A) lie inside the open unit-circle.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1), (2) or (3), (4)
%   B    - matrix with dimensions n x m in (1) or (3)
%   C    - matrix with dimensions p x n in (2) or (4)
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
%   |                 | (default 1.0e-02*sqrt(n*eps))                     |
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
%   X    - solution matrix of (1) or (2)
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
% See also ml_dlyap_smith, ml_dlyap_smith_fac.

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

narginchk(3, 5);

if (nargin < 5) || isempty(opts)
    opts = struct();
end

% Check input matrices.
n = size(A, 1);

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

if (nargin >= 4) && not(isempty(E))
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
    opts.CompTol = 1.0e-02 * sqrt(n * eps);
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
    [R, L] = deal([]);
    info   = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if hasE
    A = E \ A;
    R = E \ B;
else
    R = B;
end
L = C;

niter     = 1;
converged = 0;

[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SQUARED SMITH ITERATION.                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    % Construction of next solution matrix.
    AR = A * R;
    AL = L * A;
    R = ml_compress_fac([R, AR], opts.CompTol, 'col');
    L = ml_compress_fac([L; AL], opts.CompTol, 'row');

    % Update of iteration matrix.
    A = A * A;

    % Information about current iteration step.
    normAR = norm(AR, 'fro');
    normAL = norm(AL, 'fro');
    if normAR >= normAL
        abserr(niter) = normAR;
        relerr(niter) = abserr(niter) / norm(R, 'fro');
    else
        abserr(niter) = normAL;
        relerr(niter) = abserr(niter) / norm(L, 'fro');
    end

    if opts.Info
        fprintf(1, ['DLYAPDL_SMITH_FAC step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if hasE
    L = (L / E)';
else
    L = L';
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
