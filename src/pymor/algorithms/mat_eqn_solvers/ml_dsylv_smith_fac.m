function [Y, Z, info] = ml_dsylv_smith_fac(A, B, G, H, E, F, opts)
%ML_DSYLV_SMITH_FAC Discrete-time factorized Sylvester equation solver.
%
% SYNTAX:
%   [X, info] = ML_DSYLV_SMITH_FAC(A, B, G, H)
%   [X, info] = ML_DSYLV_SMITH_FAC(A, B, G, H, [])
%   [X, info] = ML_DSYLV_SMITH_FAC(A, B, G, H, [], [])
%   [X, info] = ML_DSYLV_SMITH_FAC(A, B, G, H, [], [], opts)
%
%   [X, info] = ML_DSYLV_SMITH_FAC(A, B, G, H, E, F)
%   [X, info] = ML_DSYLV_SMITH_FAC(A, B, G, H, E, F, opts)
%
% DESCRIPTION:
%   Computes the solution matrix of the standard discrete-time factorized
%   Sylvester equation
%
%       A*X*B - X + G*H = 0,                                            (1)
%
%   or of the generalized Sylvester equation
%
%       A*X*B - E*X*F + G*H = 0,                                        (2)
%
%   using the Smith iteration. It is assumed that the eigenvalues of A and
%   B (or s*E - A and s*F - B) lie inside the open unit disk.
%
% INPUTS:
%   A    - matrix with dimensions n x n from (1) or (2)
%   B    - matrix with dimensions m x m from (1) or (2)
%   C    - matrix with dimensions n x m from (1) or (2)
%   G    - matrix with dimensions n x p from (1) or (2)
%   H    - matrix with dimensions p x m from (1) or (2)
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
%   | CompTol         | nonnegative scalar, tolerance for the row         |
%   |                 | compression during the iteration                  |
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
%   Y    - solution factor of the Sylvester equation (1) or (2), such that
%          X = Y*Z
%   Z    - solution factor of the Sylvester equation (1) or (2), such that
%          X = Y*Z
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
%   P. Benner, Factorized solution of Sylvester equations with applications
%   in control, in: Proc. Intl. Symp. Math. Theory Networks and Syst.
%   MTNS 2004, 2004
%
% See also ml_dsylv_smith, ml_sylv_sgn_fac.

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

narginchk(4, 7);

if (nargin < 7) || isempty(opts)
    opts = struct();
end

% Check input matrices.
n = size(A, 1);
m = size(B, 1);
p = size(G, 2);

assert(isequal(size(A), [n n]), ...
    'MORLAB:data', ...
    'The matrix A has to be square!');

assert(isequal(size(B), [m m]), ...
    'MORLAB:data', ...
    'The matrix B has to be square!');

assert(isequal(size(G), [n p]), ...
    'MORLAB:data', ...
    'The matrix G must have the dimensions %d x %d!', ...
    n, p);

assert(isequal(size(H), [p m]), ...
    'MORLAB:data', ...
    'The matrix H must have the dimensions %d x %d!', ...
    p, m);

if issparse(A), A = full(A); end
if issparse(B), B = full(B); end
if issparse(G), G = full(G); end
if issparse(H), H = full(H); end

if (nargin >= 5) && not(isempty(E))
    assert(isequal(size(E), [n n]), ...
        'MORLAB:data', ...
        'The matrix E must have the same dimensions as B!');

    if issparse(E), E = full(E); end

    hasE = 1;
else
    E    = eye(n);
    hasE = 0;
end

if (nargin >= 6) && not(isempty(F))
    assert(isequal(size(F), [m m]), ...
        'MORLAB:data', ...
        'The matrix F must have the same dimensions as A!');

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
    Y    = [];
    Z    = zeros(0, m);
    info = struct([]);
    return;
elseif isempty(B)
    Y    = zeros(n, 0);
    Z    = [];
    info = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if hasE
    A = E \ A;
    Y = E \ G;
else
    Y = G;
end

if hasF
    B = B / F;
    Z = H / F;
else
    Z = H;
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
    AY = A * Y;
    ZB = Z * B;

    [Q1, R1]  = qr([Y, AY], 0);
    [Q2, R2]  = qr([Z; ZB]', 0);
    [U, S, V] = svd(R1 * R2', 'econ');

    r = sum(diag(S) > S(1) * opts.CompTol);
    Y = Q1 * (U(:, 1:r) * S(1:r, 1:r));
    Z = V(:, 1:r)' * Q2';

    % Update of iteration matrix.
    A = A * A;
    B = B * B;

    % Information about current iteration step.
    abserr(niter) = norm(AY * ZB, 'fro');
    relerr(niter) = abserr(niter) / norm(Y * Z,'fro');

    if opts.Info
        fprintf(1, ['DSYLV_SMITH_FAC step: %4d absolute error: %e' ...
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
