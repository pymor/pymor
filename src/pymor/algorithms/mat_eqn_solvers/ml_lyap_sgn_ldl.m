function [Z, Y, info] = ml_lyap_sgn_ldl(A, B, R, E, opts)
%ML_LYAP_SGN_LDL Continuous-time Lyapunov equation solver.
%
% SYNTAX:
%   [Z, Y, info] = ML_LYAP_SGN_LDL(A, B)
%   [Z, Y, info] = ML_LYAP_SGN_LDL(A, B, R)
%   [Z, Y, info] = ML_LYAP_SGN_LDL(A, B, R, [])
%   [Z, Y, info] = ML_LYAP_SGN_LDL(A, B, R, [], opts)
%
%   [Z, Y, info] = ML_LYAP_SGN_LDL(A, B, R, E)
%   [Z, Y, info] = ML_LYAP_SGN_LDL(A, B, R, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solution of the standard Lyapunov equation
%
%       A*X + X*A' + B*R*B' = 0,                                        (1)
%
%   or of the generalized Lyapunov equation
%
%       A*X*E' + E*X*A' + B*R*B' = 0,                                   (2)
%
%   with X = Z*Y*Z', using the sign function iteration. It is assumed that
%   the eigenvalues of A (or s*E - A) lie in the open left half-plane.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1) or (2)
%   B    - matrix with dimensions n x m in (1) or (2)
%   R    - symmetric matrix with dimensions m x m in (1) or (2),
%          if empty R is assumed to be the identity
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
%   Z    - full-rank solution factor of (1) or (2), such that X = Z*Y*Z'
%   Y    - full-rank solution factor of (1) or (2), such that X = Z*Y*Z'
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
% See also ml_lyapdl_sgn_ldl, ml_lyap_sgn_fac.

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

narginchk(2, 5);

if (nargin < 5) || isempty(opts)
    opts = struct();
end

% Check input matrices.
n = size(A, 1);
m = size(B, 2);

assert(isequal(size(A), [n n]), ...
    'MORLAB:data', ...
    'The matrix A has to be square!');

assert(size(B, 1) == n, ...
    'MORLAB:data', ...
    'The matrix B must have the same number of rows as A!');

if issparse(A), A = full(A); end
if issparse(B), B = full(B); end

if (nargin >= 3) && not(isempty(R))
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
    [Z, Y] = deal([]);
    info   = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Z         = B;
Y         = R;
niter     = 1;
converged = 0;

if hasE
    normE = norm(E, 'fro');
else
    normE = sqrt(n);
end

[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIGN FUNCTION ITERATION.                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    EAinv = E / A;
    if hasE
        EAinvE = EAinv * E;
    else
        EAinvE = EAinv;
    end

    % Scaling factor for convergence acceleration.
    if (niter == 1) || (relerr(niter-1) > 1.0e-02)
        c = sqrt(norm(A, 'fro') / norm(EAinvE, 'fro'));
    else
        c = 1.0;
    end
    c1 = 1.0 / (2.0 * c);
    c2 = 0.5 * c;

    % Construction of full-rank factorization with LDL row compression.
    [Z, Y] = ml_compress_ldl([Z, EAinv * Z], blkdiag(c1 * Y, c2 * Y), ...
        opts.CompTol, 'col');

    % Update of iteration matrix.
    A = c1 * A + c2 * EAinvE;

    % Information about current iteration step.
    abserr(niter) = norm(A + E, 'fro');
    relerr(niter) = abserr(niter) / normE;

    if opts.Info
        fprintf(1, ['LYAP_SGN_LDL step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if hasE
    Z = sqrt(0.5) * (E \ Z);
else
    Z = sqrt(0.5) * Z;
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

info = struct( ...
    'AbsErr'        , abserr(1:niter), ...
    'IterationSteps', niter, ...
    'RelErr'        , relerr(1:niter));
