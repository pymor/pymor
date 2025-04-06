function [X, info] = ml_dsylv_smith(A, B, C, E, F, opts)
%ML_DSYLV_SMITH Discrete-time Sylvester equation solver.
%
% SYNTAX:
%   [X, info] = ML_DSYLV_SMITH(A, B, C)
%   [X, info] = ML_DSYLV_SMITH(A, B, C, [])
%   [X, info] = ML_DSYLV_SMITH(A, B, C, [], [])
%   [X, info] = ML_DSYLV_SMITH(A, B, C, [], [], opts)
%
%   [X, info] = ML_DSYLV_SMITH(A, B, C, E, F)
%   [X, info] = ML_DSYLV_SMITH(A, B, C, E, F, opts)
%
% DESCRIPTION:
%   Computes the solution matrix of the standard discrete-time Sylvester
%   equation
%
%       A*X*B - X + C = 0,                                              (1)
%
%   or of the generalized Sylvester equation
%
%       A*X*B - E*X*F + C = 0,                                          (2)
%
%   using the Smith iteration. It is assumed that the eigenvalues of A and
%   B (or s*E - A and s*F - B) lie inside the open unit disk.
%
% INPUTS:
%   A    - matrix with dimensions n x n from (1) or (2)
%   B    - matrix with dimensions m x m from (1) or (2)
%   C    - matrix with dimensions n x m from (1) or (2)
%   E    - matrix with dimensions n x n from (2),
%          if empty E is assumed to be the identity
%   F    - matrix with dimensions m x m from (2),
%          if empty F is assumed to be the identity
%   opts - structure, containing the following optional entries:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsTol          | nonnegative scalar, tolerance for the absolute    |
%   |                 | error in the last iteration step                  |
%   |                 | (default 0)                                       |
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
%   X    - solution of the Sylvester equation (1) or (2)
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
% See also ml_dlyap_smith, ml_sylv_sgn.

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

narginchk(3, 6);

if (nargin < 6) || isempty(opts)
    opts = struct();
end

% Check input matrices.
n = size(A, 1);
m = size(B, 1);

assert(isequal(size(A), [n n]), ...
    'MORLAB:data', ...
    'The matrix A has to be square!');

assert(isequal(size(B), [m m]), ...
    'MORLAB:data', ...
    'The matrix B has to be square!');

assert(isequal(size(C), [n m]), ...
    'MORLAB:data', ...
    'The matrix C must have the dimensions %d x %d!', ...
    n, m);

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

if (nargin >= 5) && not(isempty(F))
    assert(isequal(size(F), [m m]), ...
        'MORLAB:data', ...
        'The matrix F must have the same dimensions as B!');

    if issparse(F), F = full(F); end

    hasF = 1;
else
    F    = eye(m);
    hasF = 0;
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
    opts.RelTol = 1.0e+01 * n * eps;
end

% Case of empty data.
if isempty(A)
    X    = zeros(0, m);
    info = struct([]);
    return;
elseif isempty(B)
    X    = zeros(n, 0);
    info = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if hasE
    A = E \ A;
    X = E \ C;
else
    X = C;
end

if hasF
    B = B / F;
    X = X / F;
end

niter     = 1;
converged = 0;

[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SQUARED SMITH ITERATION.                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    % Construction of next solution matrix.
    AXB = A * (X * B);
    X   = X + AXB;

    % Update of iteration matrix.
    A = A * A;
    B = B * B;

    % Information about current iteration step.
    abserr(niter) = norm(AXB, 'fro');
    relerr(niter) = abserr(niter) / norm(X,'fro');

    if opts.Info
        fprintf(1, ['DSYLV_SMITH step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
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
