function [Z, info] = ml_icare_ric_fac(A, B1, B2, C1, E, opts)
%ML_ICARE_RIC_FAC Continuous-time indefinite Riccati equation solver.
%
% SYNTAX:
%   [Z, info] = ML_ICARE_RIC_FAC(A, B1, B2, C1)
%   [Z, info] = ML_ICARE_RIC_FAC(A, B1, B2, C1, [])
%   [Z, info] = ML_ICARE_RIC_FAC(A, B1, B2, C1, [], opts)
%
%   [Z, info] = ML_ICARE_RIC_FAC(A, B1, B2, C1, E)
%   [Z, info] = ML_ICARE_RIC_FAC(A, B1, B2, C1, E, opts)
%
% DESCRIPTION:
%   Computes the low-rank solutions of the standard Riccati equation with
%   indefinite quadratic term
%
%       A'*X + X*A + X*(B1*B1' - B2*B2')*X + C1'*C1 = 0                 (1)
%
%   or of the generalized Riccati equation with indefinite quadratic term
%
%       A'*X*E + E'*X*A + E'*X*(B1*B1' - B2*B2')*X*E + C1'*C1 = 0       (2)
%
%   with X = Z*Z', using the Low-Rank Riccati iteration. It is assumed that
%   the eigenvalues of A (or s*E - A) lie in the open left half-plane.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1) or (2)
%   B1   - matrix with dimensions n x m1 in (1) or (2)
%   B2   - matrix with dimensions n x m2 in (1) or (2)
%   C    - matrix with dimensions p x n in (1) or (2)
%   E    - matrix with dimensions n x n in (2),
%          if empty the standard equation (1) is solved
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
%   |                 | (default 1.0e-02*sqrt(n*eps))                     |
%   +-----------------+---------------------------------------------------+
%   | Info            | {0, 1}, used to disable/enable display of verbose |
%   |                 | status information during iteration steps         |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | careopts        | structure, containing the optional parameters for |
%   |                 | the computation of the continuous-time definite   |
%   |                 | Riccati equations used in every iteration step,   |
%   |                 | see ml_caredl_sgn_fac                             |
%   |                 | (default struct())                                |
%   +-----------------+---------------------------------------------------+
%   | MaxIter         | positive integer, maximum number of iteration     |
%   |                 | steps                                             |
%   |                 | (default 100)                                     |
%   +-----------------+---------------------------------------------------+
%   | RelTol          | nonnegative scalar, tolerance for the relative    |
%   |                 | error in the last iteration step                  |
%   |                 | (default 1.0e+02*n*eps)                           |
%   +-----------------+---------------------------------------------------+
%   | StoreLQG        | {0, 1}, used to disable/enable storing of         |
%   |                 | low-rank solution factor to the corresponding LQG |
%   |                 | problem                                           |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | Z0              | matrix of dimensions n x k, solution factor of    |
%   |                 | the corresponding LQG problem such that X = Z*Z', |
%   |                 | A'XE + E'XA - E'XB2B2'XE + C'C = 0                |
%   |                 | (default [])                                      |
%   +-----------------+---------------------------------------------------+
%
% OUTPUTS:
%   Z    - low-rank solution factor of (1) or (2), such that X = Z*Z'
%   info - structure, containing the following information:
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsErr          | vector, containing the absolute error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%   | infoCARE        | array of structs, containing information about    |
%   |                 | the used Riccati equations solver for every       |
%   |                 | iteration step, see ml_care_sgn_fac               |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the relative error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%   | Z_LQG           | low-rank solution factor of the corresponding     |
%   |                 | LQG problem                                       |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%   A. Lanzon, Y. Feng, B. D. O. Anderson, An iterative algorithm to solve
%   algebraic Riccati equations with an indefinite quadratic term, in:
%   2007 European Control Conference (ECC), 2007, pp. 3033--3039.
%   https://doi.org/10.23919/ecc.2007.7068239
%
% See also ml_care_nwt_fac, ml_pcare_nwt_fac.

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

narginchk(4, 6);

if (nargin < 6) || (isempty(opts))
    opts = struct();
end

% Check input matrices.
n = size(A, 1);

assert(isequal(size(A), [n n]), ...
    'MORLAB:data', ...
    'The matrix A has to be square!');

assert(size(B1, 1) == n, ...
    'MORLAB:data', ...
    'The matrix B1 must have the same number of rows as A!');

assert(size(B2, 1) == n, ...
    'MORLAB:data', ...
    'The matrix B2 must have the same number of rows as A!');

assert(size(C1, 2) == n, ...
    'MORLAB:data', ...
    'The matrix C must have the same number of columns as A!');

if issparse(A), A = full(A); end
if issparse(B1), B1 = full(B1); end
if issparse(B2), B2 = full(B2); end
if issparse(C1), C1 = full(C1); end

if (nargin > 4) && not(isempty(E))
    assert(isequal(size(E), [n n]), ...
        'MORLAB:data', ...
        'The matrix E must have the same dimensions as A!');

    if issparse(E), E = full(E); end

    hasE = 1;
else
    E    = [];
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

if ml_field_set_to_value(opts, 'careopts')
    assert(isa(opts.careopts, 'struct'), ...
        'MORLAB:data', ...
        'The parameter opts.careopts has to be a struct!');
else
    opts.careopts = struct();
end
opts.careopts.EqnType = 'primal';

if ml_field_set_to_value(opts, 'Info')
    ml_assert_boolean(opts.Info, 'opts.Info');
else
    opts.Info = false;
end

if ml_field_set_to_value(opts, 'CompTol')
    ml_assert_nonnegscalar(opts.CompTol, 'opts.CompTol');
else
    opts.CompTol = 1.0e-02 * sqrt(n * eps);
end

if ml_field_set_to_value(opts, 'MaxIter')
    ml_assert_posinteger(opts.MaxIter, 'opts.MaxIter');
else
    opts.MaxIter = 100;
end

if ml_field_set_to_value(opts, 'RelTol')
    ml_assert_nonnegscalar(opts.RelTol, 'opts.RelTol');
else
    opts.RelTol = 1.0e+02 * (n * eps);
end

if ml_field_set_to_value(opts, 'StoreLQG')
    ml_assert_boolean(opts.StoreLQG, 'opts.StoreLQG');
else
    opts.StoreLQG = false;
end

% Case of empty data.
if isempty(A)
    Z    = [];
    info = struct([]);
    return;
end

% Initial info structure.
info = struct();


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Solve the initial LQG problem.
if ml_field_set_to_value(opts, 'Z0')
    assert(isa(opts.Z0, 'double') && (size(opts.Z0, 1) == n), ...
        'MORLAB:data', ...
        'The initial solution matrix Z0 must have %d columns!', ...
        n);

    Y = full(opts.Z0);
else
    % Solve the initial LQG problem.
    [Y, ~, infoCARE] = ml_caredl_sgn_fac(A, B2, C1, E, -1, opts.careopts);

    if nargout == 2
        info.Z_LQG       = Y;
        info.infoCARE(1) = infoCARE;
    end
end

% Initialization.
niter            = 1;
converged        = 0;
Z                = zeros(n, 0);
[abserr, relerr] = deal(zeros(1, opts.MaxIter));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Low-Rank Riccati Iteration.                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while niter <= opts.MaxIter
    if opts.CompTol && (niter > 1)
        % Construction of next solution factor with row compression.
        Z = ml_compress_fac([Z, Y], opts.CompTol, 'col');
    else
        Z(:, end+1:end+size(Y, 2)) = Y;
    end

    % Update the residual term.
    W = (B1' * Y) * Y';
    if hasE
        W = W * E;
    end

    % Compute convergence measures.
    abserr(niter) = norm(W * W', 'fro');
    relerr(niter) = abserr(niter) / max(norm(Z' * Z, 'fro'), 1);

    % Information about current iteration step.
    if opts.Info
        fprintf(1, ['ICARE_RIC_FAC step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Evaluate stopping criteria.
    if (abserr(niter) <= opts.AbsTol) ...
            || (relerr(niter) <= opts.RelTol) ...
            || (niter >= opts.MaxIter)
        if (abserr(niter) <= opts.AbsTol) ...
                || (relerr(niter) <= opts.RelTol)
            converged = 1;
        end
        break;
    end

    % Solve the next residual equation.
    if hasE
        [Y, ~, infoCARE] = ml_caredl_sgn_fac(A + B1 * (((B1' * Z) * Z') * E) ...
            - B2 * (((B2' * Z) * Z') * E), B2, W, E, -1, opts.careopts);
    else
        [Y, ~, infoCARE] = ml_caredl_sgn_fac(A + B1 * ((B1' * Z) * Z') ...
            - B2 * ((B2' * Z) * Z'), B2, W, [], -1, opts.careopts);
    end

    niter = niter + 1;

    if nargout == 2
        info.infoCARE(niter) = infoCARE;
    end
end

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
info.AbsErr         = abserr(1:niter);
info.IterationSteps = niter;
info.RelErr         = relerr(1:niter);

[~, perm] = sort(lower(fieldnames(info)));
info      = orderfields(info, perm);
