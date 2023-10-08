function [Z, info] = ml_gdlyap_smith_fac(A, B, E, opts)
%ML_GDLYAP_SMITH_FAC Nilpotent discrete-time Lyapunov equation solver.
%
% SYNTAX:
%   [Z, info] = ML_GDLYAP_SMITH_FAC(A, B, C, E)
%   [Z, info] = ML_GDLYAP_SMITH_FAC(A, B, C, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solutions X = Z*Z' of the generalized
%   discrete-time dual Lyapunov equations
%
%       E*X*E' - A*X*A' + B*B' = 0,                                     (1)
%
%   where E is nilpotent and A is invertible, i.e., the matrix pencil
%   s*E - A has only infinite eigenvalues, via the Smith iteration.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1)
%   B    - matrix with dimensions n x m in (1)
%   E    - matrix with dimensions n x n in (1)
%   opts - structure, containing the following optional entries:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsTol          | nonnegative scalar, tolerance for the absolute    |
%   |                 | gain in the last iteration step                   |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | Index           | nonnegative integer, index of nilpotency of the   |
%   |                 | matrix E used to set the exact number of iteration|
%   |                 | steps, if the index is unknown Inf is set         |
%   |                 | (default Inf)                                     |
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
%   |                 | gain in the last iteration step                   |
%   |                 | (default 1.0e+01*n*eps)                           |
%   +-----------------+---------------------------------------------------+
%
% OUTPUTS:
%   Z    - full-rank factor of (1), such that X = Z*Z'
%   info - structure, containing the following information
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsErr          | vector, containing the absolute change of the     |
%   |                 | iteration matrix in each iteration step           |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the relative change of the     |
%   |                 | iteration matrix in each iteration step           |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%   T. Stykel, Low-rank iterative methods for projected generalized
%   Lyapunov equations, Electron. Trans. Numer. Anal. 30 (2008) 187--202.
%
% See also ml_gdlyapdl_smith_fac, ml_lyap_sgn_fac.

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

narginchk(3, 4);

if (nargin < 4) || isempty(opts)
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

assert(isequal(size(E), [n n]), ...
    'MORLAB:data', ...
    'The matrix E must have the same dimensions as A!');

if issparse(B), B = full(B); end

% Check and assign optional parameters.
assert(isa(opts, 'struct'), ...
    'MORLAB:data', ...
    'The parameter opts has to be a struct!');

if ml_field_set_to_value(opts, 'AbsTol')
    ml_assert_nonnegscalar(opts.AbsTol, 'opts.AbsTol');
else
    opts.AbsTol = 0;
end

if ml_field_set_to_value(opts, 'Index')
    ml_assert_nonnegintinf(opts.Index, 'opts.Index');
else
    opts.Index = Inf;
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
    Z    = [];
    info = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Catch index-0 case.
if opts.Index == 0
    Z    = zeros(n, 0);
    info = struct(...
        'AbsErr'        , [], ...
        'IterationSteps', 0, ...
        'RelErr'        , []);

    return;
end

maxiter = min(opts.Index, opts.MaxIter);
Z       = zeros(n, maxiter * m);
niter   = 2;

[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIRST STEP.                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[AL, AU, q] = lu(A, 'vector');
Y           = AU \ (AL \ B(q, :));
Z(:, 1:m)   = Y;

abserr(1) = norm(Y, 'fro');
relerr(1) = 1.0;

if opts.Info
    fprintf(1, ['GDLYAP_SMITH_FAC step: %4d absolute change: ' ...
        '%e relative change: %e \n'], ...
        niter, abserr(1), relerr(1));
end

% Method is converged if absolute or relative changes are small enough.
converged = (abserr(1) <= opts.AbsTol) || (relerr(1) <= opts.RelTol);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SMITH ITERATION.                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= maxiter) && not(converged)
    Y                           = AU \ (AL \ (E(q, :) * Y));
    Z(:, (niter-1)*m+1:niter*m) = Y;

    normY = norm(Y, 'fro');

    abserr(niter) = normY;
    relerr(niter) = normY / norm(Z(:, 1:niter*m), 'fro');

    % Information about current iteration step.
    if opts.Info
        fprintf(1, ['GDLYAP_SMITH_FAC step: %4d absolute change: ' ...
            '%e relative change: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative changes are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

niter = niter - 1;

% Cut off unnecessary iteration step.
if converged
    Z = Z(:, 1:m*(niter-1));
end

% Warning if iteration not converged.
if (niter == opts.MaxIter) && not(converged)
    warning('MORLAB:noConvergence', ...
        ['No convergence in %d iteration steps!\n' ...
        'Abs. tolerance: %e | Abs. change: %e\n' ...
        'Rel. tolerance: %e | Rel. change: %e\n' ...
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
