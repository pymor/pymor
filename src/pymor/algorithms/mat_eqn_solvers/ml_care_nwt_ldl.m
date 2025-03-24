function [L, D, info] = ml_care_nwt_ldl(A, B, C, Q, R, S, E, opts)
%ML_CARE_NWT_LDL Continuous-time factorized Riccati equation solver.
%
% SYNTAX:
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C)
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, [])
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, [], [])
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, [], [], [])
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, [], [], [], [])
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, [], [], [], [], opts)
%
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, Q, [], [], [], opts)
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, Q, R, [], [], opts)
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, Q, R, S, [], opts)
%   [L, D, info] = ML_CARE_NWT_LDL(A, B, C, Q, R, S, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solutions of the standard algebraic Riccati
%   equation
%
%       A'*X*E + E'*X*A + C'*Q*C - (E'*X*B + S)*R^-1*(B'*X*E + S') = 0, (1)
%
%   with X = L*D*L', using the Newton-Kleinman iteration.
%   It is assumed that the eigenvalues of A (or s*E - A) lie in the open
%   left half-plane, otherwise a stabilizing initial feedback K0 must be
%   given as parameter.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1)
%   B    - matrix with dimensions n x m in (1)
%   C    - matrix with dimensions p x n in (1)
%   Q    - symmetric matrix with dimensions p x p in (1)
%          if empty set to identity
%   R    - invertible, symmetric matrix with dimensions m x m in (1),
%          if empty set to identity
%   S    - matrix with dimensions n x m in (1),
%          if empty set ot zero
%   E    - invertible matrix with dimensions n x n in (1),
%          if empty set to identity
%   opts - structure, containing the following optional entries:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsTol          | nonnegative scalar, tolerance for the absolute    |
%   |                 | residual in the last iteration step               |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | Info            | {0, 1}, used to disable/enable display of verbose |
%   |                 | status information during iteration steps         |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | K0              | matrix with dimensions m x n, used to stabilize   |
%   |                 | the spectrum of s*E - A, such that s*E-(A-B*K0)   |
%   |                 | has only stable eigenvalues                       |
%   |                 | (default zeros(m, n))                             |
%   +-----------------+---------------------------------------------------+
%   | lyapopts        | structure, containing the optional parameters for |
%   |                 | Lyapunov equation solver, see ml_lyap_sgn_ldl     |
%   |                 | (default struct())                                |
%   +-----------------+---------------------------------------------------+
%   | MaxIter         | positive integer, maximum number of iteration     |
%   |                 | steps                                             |
%   |                 | (default 100)                                     |
%   +-----------------+---------------------------------------------------+
%   | RelTol          | nonnegative scalar, tolerance for the normalized  |
%   |                 | residual in the last iteration step               |
%   |                 | (default 1.0e+02*n*eps)                           |
%   +-----------------+---------------------------------------------------+
%
% OUTPUTS:
%   L    - full-rank solution factor of (1), such that X = L*D*L'
%   D    - full-rank solution factor of (1), such that X = L*D*L'
%   info - structure, containing the following information:
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsErr          | vector, containing the absolute residuals of the  |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%   | infoLYAP        | array of structs, containing information about    |
%   |                 | the used Lyapunov equations solver for every      |
%   |                 | iteration step, see ml_lyap_sgn_fac               |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the normalized residuals of the|
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%
% See also ml_icare_ric_fac, ml_pcare_nwt_fac.

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

narginchk(3, 8);

if (nargin < 8) || isempty(opts)
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

if (nargin >= 4) && not(isempty(Q))
    assert(isequal(size(Q), [p p]), ...
        'MORLAB:data', ...
        'The matrix Q must be square with the number of rows as C!');

    assert(norm(Q - Q', 'fro') < 1.0e-14, ...
        'MORLAB:data', ...
        'The matrix Q must be symmetric!');

    if issparse(Q), Q = full(Q); end

    hasQ = 1;
else
    Q    = eye(p);
    hasQ = 0;
end

if (nargin >= 5) && not(isempty(R))
    assert(isequal(size(R), [m, m]), ...
        'MORLAB:data', ...
        'The matrix R must be square with the number of columns as B!');

    assert(norm(R - R', 'fro') < 1.0e-14, ...
        'MORLAB:data', ...
        'The matrix R must be symmetric!');

    if issparse(R), R = full(R); end

    hasR = 1;
else
    R    = eye(m);
    hasR = 0;
end

if (nargin >= 6) && not(isempty(S))
    assert(isequal(size(S), [n, m]), ...
        'MORLAB:data', ...
        'The matrix S must have the same dimensions as B!');

    if issparse(S), S = full(S); end
else
    S = zeros(size(B));
end

if (nargin >= 7) && not(isempty(E))
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

if ml_field_set_to_value(opts, 'Info')
    ml_assert_boolean(opts.Info, 'opts.Info');
else
    opts.Info = false;
end

if ml_field_set_to_value(opts, 'K0')
    assert(isa(opts.K0, 'double') && isequal(size(opts.K0), [m n]), ...
        'MORLAB:data', ...
        'The stabilization matrix K0 must have the dimensions %d x %d!',...
        m, n);

    K = opts.K0;
else
    K = zeros(m, n);
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
    L    = [];
    D    = [];
    info = struct([]);
    return;
end

% Initial info structure.
info = struct();


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

niter            = 1;
converged        = 0;
[absres, nrmres] = deal(zeros(1, opts.MaxIter));

% Center matrix in RHS.
T = blkdiag(Q, R, -R);

% Right-hand side.
if hasQ
    CQC = C' * (Q * C);
else
    CQC = C' * C;
end

nrmRhs = norm(CQC, 'fro');

% Mixing terms.
if hasR
    RS = R \ S';
else
    RS = S';
end
RBXE = K - RS;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEWTON-KLEINMAN ITERATION.                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    W                = [C; RBXE; RS];
    [L, D, infoLYAP] = ml_lyap_sgn_ldl((A - B * K)', W', T, E', ...
        opts.lyapopts);

    if hasE
        LE = L' * E;
    else
        LE = L';
    end
    AL = A' * (L * D);

    if hasR
        RBXE = R \ (((B' * L) * D) * LE);
    else
        RBXE = ((B' * L) * D) * LE;
    end
    K = RBXE + RS;

    % Information about current iteration step.
    absres(niter) = norm(AL * LE + LE' * AL' - K' * (R * K) + CQC, 'fro');
    nrmres(niter) = absres(niter) / nrmRhs;

    if opts.Info
        fprintf(1, ['CARE_NWT_LDL step: %4d absolute residual: %e' ...
            ' normalized residual: %e \n'], ...
            niter, absres(niter), nrmres(niter));
    end

    % Save Lyapunov solver information.
    if nargout == 2
        info.infoLYAP(niter) = infoLYAP;
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (absres(niter) <= opts.AbsTol) || ...
        (nrmres(niter) <= opts.RelTol);
    niter     = niter + 1;
end

niter = niter - 1;

% Warning if iteration not converged.
if (niter == opts.MaxIter) && not(converged)
    warning('MORLAB:noConvergence', ...
        ['No convergence in %d iteration steps!\n' ...
        'Abs. tolerance: %e | Abs. residual: %e\n' ...
        'Rel. tolerance: %e | Rel. residual: %e\n' ...
        'Try to increase the tolerances or number of ' ...
        'iteration steps!'], ...
        niter, opts.AbsTol, absres(niter), ...
        opts.RelTol, nrmres(niter));
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASSIGN INFORMATION.                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assign information about iteration.
info.AbsErr         = absres(1:niter);
info.IterationSteps = niter;
info.RelErr         = nrmres(1:niter);

[~, perm] = sort(lower(fieldnames(info)));
info      = orderfields(info, perm);
