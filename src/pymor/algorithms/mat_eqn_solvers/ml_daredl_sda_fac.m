function [R, L, info] = ml_daredl_sda_fac(A, B, C, E, alpha, opts)
%ML_DAREDL_SDA_FAC Discrete-time dual Riccati equation solver.
%
% SYNTAX:
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C)
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C, [])
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C, [], alpha)
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C, [], alpha, opts)
%
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C, E, [])
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C, E, alpha)
%   [R, L, info] = ML_DAREDL_SDA_FAC(A, B, C, E, alpha, opts)
%
% DESCRIPTION:
%   Computes the solution factors of the dual standard discrete-time
%   Riccati equations
%
%       A'*X*A - X + alpha * A'*X*B*inv(I - alpha * B'*X*B)*B'*X*A
%           + C'*C = 0,                                                 (1)
%       A*Y*A' - Y + alpha * A*Y*C'*inv(I - alpha * C*Y*C')*C*Y*A'
%           + B*B' = 0,                                                 (2)
%
%   or of the generalized Riccati equations
%
%       A'*X*A - E'*X*E + alpha * A'*X*B*inv(I - alpha * B'*X*B)*B'*X*A
%           + C'*C = 0,                                                 (3)
%       A*Y*A' - E'*Y*E + alpha * A*Y*C'*inv(I - alpha * C*Y*C')*C*Y*A
%           + B*B' = 0,                                                 (4)
%
%   using the structure-preserving doubling algorithm, such that X = R*R'
%   and Y = L*L'. It is assumed that E is invertible.
%
% INPUTS:
%   A     - matrix with dimensions n x n in (1), (2) or (3), (4)
%   B     - matrix with dimensions n x m in (1), (2) or (3), (4)
%   C     - matrix with dimensions p x n in (1), (2) or (3), (4)
%   E     - matrix with dimensions n x n in (3), (4),
%           if empty the standard equations (1), (2) are solved
%   alpha - real scalar, scaling of the quadratic term,
%           if empty alpha is assumed to be -1
%   opts  - structure, containing the following optional entries:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsTol          | nonnegative scalar, tolerance for the absolute    |
%   |                 | error in the last iteration step                  |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | CompTol         | nonnegative scalar, tolerance for the row and     |
%   |                 | column compression during the iteration           |
%   |                 | (default 1.0e-02*sqrt(n*eps))                     |
%   +-----------------+---------------------------------------------------+
%   | EqnType         | character array, switch for only computing the    |
%   |                 | solution of one type of the above equations       |
%   |                 |  'primal' - equation (1) or (3) is solved         |
%   |                 |  'dual'   - equation (2) or (4) is solved         |
%   |                 |  'both'   - both equations are solved             |
%   |                 | (default 'both')                                  |
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
%   |                 | (default 1.0e+02*n*eps)                           |
%   +-----------------+---------------------------------------------------+
%
% OUTPUTS:
%   R    - full-rank solution factor of (1) or (3), s.t. X = R*R'
%   L    - full-rank solution factor of (2) or (4), s.t. Y = L*L'
%   info - structure, containing the following information:
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsErr          | vector, containing the absolute error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the relative error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%   E. K.-W. Chu, H.-Y. Fan, W.-W. Lin, C.-S. Wang, Structure-preserving
%   algorithms for periodic discrete-time algebraic Riccati equations,
%   Internat. J. Control 77 (8) (2004) 767--788.
%
% See also ml_daredl_sda, ml_caredl_sgn_fac.

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

if (nargin >= 5) && not(isempty(alpha))
    ml_assert_scalar(alpha, 'alpha');
else
    alpha = -1;
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

if ml_field_set_to_value(opts, 'EqnType')
    assert(strcmpi(opts.EqnType, 'primal') ...
        || strcmpi(opts.EqnType, 'dual') ...
        || strcmpi(opts.EqnType, 'both'), ...
        'MORLAB:data', ...
        'The given type of equation is not implemented!');
else
    opts.EqnType = 'both';
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
    opts.RelTol = 1.0e+02 * (n * eps);
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
    A = A / E;
    C = C / E;
end

nrmA = norm(A, 'fro');

niter            = 1;
converged        = 0;
[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STRUCTURE-PRESERVING DOUBLING ALGORITHM.                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    m = size(B, 2);
    p = size(C, 1);

    CB   = C * B;
    R1   = chol(eye(m) - (alpha * CB') * CB, 'upper');
    R2   = chol(eye(p) - (alpha * CB) * CB', 'lower');
    ABR1 = (A * B) / R1;
    CA   = C * A;

    % Update of iteration matrices.
    B = ml_compress_fac([B, ABR1], opts.CompTol, 'col');
    C = ml_compress_fac([C; R2 \ CA], opts.CompTol, 'row');
    A = A * A + alpha * ((ABR1 * (CB / R1)') * CA);

    % Information about current iteration step.
    abserr(niter) = norm(A, 'fro');
    relerr(niter) = abserr(niter) / nrmA;

    if opts.Info
        fprintf(1, ['DAREDL_SDA_FAC step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if strcmpi(opts.EqnType, 'primal') || strcmpi(opts.EqnType, 'both')
    R = C';
else
    R = [];
end

if strcmpi(opts.EqnType, 'dual') || strcmpi(opts.EqnType, 'both')
    if hasE
        L = E \ B;
    else
        L = B;
    end
else
    L = [];
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
