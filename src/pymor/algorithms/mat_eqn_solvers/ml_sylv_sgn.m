function [X, info] = ml_sylv_sgn(A, B, C, E, F, opts)
%ML_SYLV_SGN Continuous-time Sylvester equation solver.
%
% SYNTAX:
%   [X, info] = ML_SYLV_SGN(A, B, C)
%   [X, info] = ML_SYLV_SGN(A, B, C, [])
%   [X, info] = ML_SYLV_SGN(A, B, C, [], [])
%   [X, info] = ML_SYLV_SGN(A, B, C, [], [], opts)
%
%   [X, info] = ML_SYLV_SGN(A, B, C, E, F)
%   [X, info] = ML_SYLV_SGN(A, B, C, E, F, opts)
%
% DESCRIPTION:
%   Computes the solution matrix of the standard Sylvester equation
%
%       A*X + X*B + C = 0,                                              (1)
%
%   or of the generalized Sylvester equation
%
%       A*X*F + E*X*B + C = 0,                                          (2)
%
%   using the sign function iteration. It is assumed that the eigenvalues
%   of A and B (or s*E - A and s*F - B) lie in the open left half-plane.
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
%   P. Benner, Factorized solution of Sylvester equations with applications
%   in control, in: Proc. Intl. Symp. Math. Theory Networks and Syst.
%   MTNS 2004, 2004
%
% See also ml_lyap_sgn, ml_cabe_sgn.

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
    opts.RelTol = 1.0e+01 * (n * eps);
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

X         = C;
niter     = 1;
converged = 0;

[abserr, relerr] = deal(zeros(1, opts.MaxIter));

normE = norm(E, 'fro');
normF = norm(F, 'fro');


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGN FUNCTION ITERATION.                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    EAinv = E / A;
    BinvF = B \ F;
    W     = EAinv * (X * BinvF);

    if hasF
        EAinv = EAinv * E;
    end

    if hasE
        BinvF = F * BinvF;
    end

    % Scaling factor for convergence acceleration.
    if (niter == 1) || (relerr(niter-1) > 1.0e-02)
        e = sqrt(norm(A, 'fro')^2 + norm(X, 'fro')^2 + norm(B, 'fro')^2);
        d = sqrt(norm(EAinv, 'fro')^2 + norm(W, 'fro')^2 + ...
            norm(BinvF, 'fro')^2);
        c = sqrt(e / d);
    else
        c = 1.0;
    end
    c1 = 1.0 / (2.0 * c);
    c2 = 0.5 * c;

    % Update of iteration matrices.
    A = c1 * A + c2 * EAinv;
    B = c1 * B + c2 * BinvF;
    X = c1 * X + c2 * W;

    % Information about current iteration step.
    Aerr          = norm(A + E, 'fro');
    Berr          = norm(B + F, 'fro');
    abserr(niter) = max([Aerr, Berr]);
    relerr(niter) = max([Aerr / normE, Berr / normF]);

    if opts.Info
        fprintf(1, ['SYLV_SGN step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

X = 0.5 * (E \ (X / F));

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
