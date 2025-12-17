function [X, Y, info] = ml_daredl_sda(A, G, F, E, alpha, opts)
%ML_DAREDL_SDA Discrete-time dual Riccati equation solver.
%
% SYNTAX:
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F)
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F, [])
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F, [], alpha)
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F, [], alpha, opts)
%
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F, E, [])
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F, E, alpha)
%   [X, Y, info] = ML_DAREDL_SDA(A, G, F, E, alpha, opts)
%
% DESCRIPTION:
%   Computes the solution factors of the dual standard discrete-time
%   Riccati equations
%
%       A'*X*A - X + alpha * A'*X*inv(I - alpha * G*X)*G*X*A + F = 0,   (1)
%       A*Y*A' - Y + alpha * A*Y*inv(I - alpha * F*Y)*F*Y*A' + G = 0,   (2)
%
%   or of the generalized Riccati equations
%
%       A'*X*A - E'*X*E + alpha * A'*X*inv(I - alpha * G*X)*G*X*A
%           + F = 0,                                                    (3)
%       A*Y*A' - E'*Y*E + alpha * A*Y*inv(I - alpha * F*Y)*F*Y*A'
%           + G = 0,                                                    (4)
%
%   using the structure-preserving doubling algorithm. It is assumed that
%   E is invertible.
%
% INPUTS:
%   A     - matrix with dimensions n x n in (1), (2) or (3), (4)
%   G     - symmetric matrix with dimensions n x n in (1), (2) or (3), (4)
%   F     - symmetric matrix with dimensions n x n in (1), (2) or (3), (4)
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
%   X    - solution of (1) or (3)
%   Y    - solution of (2) or (4)
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
% See also ml_daredl_sda_fac, ml_caredl_sgn.

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

assert(isequal(size(G), [n, n]), ...
    'MORLAB:data', ...
    'The matrix G must have the same dimensions as A!');

assert(norm(G - G', 'fro') < 1.0e-14, ...
    'MORLAB:data', ...
    'The matrix G must be symmetric!');

assert(isequal(size(F), [n, n]), ...
    'MORLAB:data', ...
    'The matrix F must have the same dimensions as A!');

assert(norm(F - F', 'fro') < 1.0e-14, ...
    'MORLAB:data', ...
    'The matrix F must be symmetric!');

if issparse(A), A = full(A); end
if issparse(G), G = full(G); end
if issparse(F), F = full(F); end

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
    [X, Y] = deal([]);
    info   = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if hasE
    A = A / E;
    F = (E' \ F) / E;
end

In   = eye(n);
nrmA = norm(A, 'fro');

niter            = 1;
converged        = 0;
[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STRUCTURE-PRESERVING DOUBLING ALGORITHM.                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    IQR  = (In - (alpha * G) * F) \ In;
    AIQR = A * IQR;

    % Update of iteration matrices.
    G = G + AIQR * (G * A');
    F = F + A' * (F * (IQR * A));
    A = AIQR * A;

    % Information about current iteration step.
    abserr(niter) = norm(A, 'fro');
    relerr(niter) = abserr(niter) / nrmA;

    if opts.Info
        fprintf(1, ['DAREDL_SDA step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if strcmpi(opts.EqnType, 'primal') || strcmpi(opts.EqnType, 'both')
    X = F;
else
    X = [];
end

if strcmpi(opts.EqnType, 'dual') || strcmpi(opts.EqnType, 'both')
    if hasE
        Y = (E \ G) / E';
    else
        Y = G;
    end
else
    Y = [];
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
