function [Z, info] = ml_pcare_nwt_fac(A, B, C, E, opts)
%ML_PCARE_NWT_FAC Positive continuous-time Riccati equation solver.
%
% SYNTAX:
%   [Z, info] = ML_PCARE_NWT_FAC(A, B, C)
%   [Z, info] = ML_PCARE_NWT_FAC(A, B, C, [])
%   [Z, info] = ML_PCARE_NWT_FAC(A, B, C, [], opts)
%
%   [Z, info] = ML_PCARE_NWT_FAC(A, B, C, E)
%   [Z, info] = ML_PCARE_NWT_FAC(A, B, C, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solutions of the standard positive Riccati
%   equation
%
%       A'*X + X*A + X*B*B'*X + C'*C = 0,                               (1)
%
%   or of the generalized positive Riccati equation
%
%       A'*X*E + E'*X*A + E'*X*B*B'*X*E + C'*C = 0,                     (2)
%
%   with X = Z*Z', using the low-rank Newton iteration. It is assumed that
%   the eigenvalues of A (or s*E - A) lie in the open left half-plane and
%   that the equation (1) (or (2)) has a solution.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1) or (2)
%   B    - matrix with dimensions n x m in (1) or (2)
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
%   | lyapopts        | structure, containing the optional parameters for |
%   |                 | the Lyapunov equation solver used in every        |
%   |                 | iteration step, see ml_lyap_sgn_fac               |
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
%   | infoLYAP        | array of structs, containing information about    |
%   |                 | the used Lyapunov equations solver for every      |
%   |                 | iteration step, see ml_lyap_sgn_fac               |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the relative error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%   A. Varga, On computing high accuracy solutions of a class of Riccati
%   equations, Control-Theory and Advanced Technology 10 (4) (1995)
%   2005--2016.
%
% See also ml_care_nwt_fac, ml_icare_ric_fac.

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

if ml_field_set_to_value(opts, 'lyapopts')
    assert(isa(opts.lyapopts, 'struct'), ...
        'MORLAB:data', ...
        'The parameter opts.lyapopts has to be a struct!');
else
    opts.lyapopts = struct();
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

niter            = 2;
[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIAL STEP.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[N, infoLYAP] = ml_lyap_sgn_fac(A', C', E', opts.lyapopts);

Z = N;
K = (B' * N) * N';
if hasE
    K = K * E;
end

abserr(1) = norm(K * K', 'fro');
relerr(1) = abserr(1) / max(norm(Z' * Z, 'fro'), 1);

if opts.Info
    fprintf(1, ['PCARE_NWT_FAC step: %4d absolute error: %e' ...
        ' relative error: %e \n'], ...
        1, abserr(1), relerr(1));
end

if nargout == 2
    info.infoLYAP(1) = infoLYAP;
end

% Method is converged if absolute or relative errors are small enough.
converged = (abserr(1) <= opts.AbsTol) || (relerr(1) <= opts.RelTol);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEWTON ITERATION.                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    if hasE
        N = ml_lyap_sgn_fac((A + B * (((B' * Z) * Z') * E))', K', E', ...
            opts.lyapopts);
    else
        N = ml_lyap_sgn_fac((A + B * ((B' * Z) * Z'))', K', [], ...
            opts.lyapopts);
    end
    K = (B' * N) * N';
    if hasE
        K = K * E;
    end

    % Row compression.
    Z = ml_compress_fac([Z, N], opts.CompTol, 'col');

    % Information about current iteration step.
    abserr(niter) = norm(K * K', 'fro');
    relerr(niter) = abserr(niter) / max(norm(Z' * Z, 'fro'), 1);

    if opts.Info
        fprintf(1, ['PCARE_NWT_FAC step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Save Lyapunov solver information.
    if nargout == 2
        info.infoLYAP(niter) = infoLYAP;
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
info.AbsErr         = abserr(1:niter);
info.IterationSteps = niter;
info.RelErr         = relerr(1:niter);

[~, perm] = sort(lower(fieldnames(info)));
info      = orderfields(info, perm);
