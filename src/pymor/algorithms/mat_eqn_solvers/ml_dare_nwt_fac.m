function [Z, info] = ml_dare_nwt_fac(A, B, C, E, opts)
%ML_DARE_NWT_FAC Discrete-time Riccati equation solver.
%
% SYNTAX:
%   [Z, info] = ML_DARE_NWT_FAC(A, B, C)
%   [Z, info] = ML_DARE_NWT_FAC(A, B, C, [])
%   [Z, info] = ML_DARE_NWT_FAC(A, B, C, [], opts)
%
%   [Z, info] = ML_DARE_NWT_FAC(A, B, C, E)
%   [Z, info] = ML_DARE_NWT_FAC(A, B, C, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solutions of the standard discrete-time
%   Riccati equation
%
%       A'*X*A - X - A'*X*B*inv(I + B'*X*B)*B'*X*A + C'*C = 0,          (1)
%
%   or of the generalized Riccati equation
%
%       A'*X*A - E'*X*E - A'*X*B*inv(I + B'*X*B)*B'*X*A + C'*C = 0,     (2)
%
%   with X = Z*Z', using the Newton-Kleinman iteration. It is assumed that
%   the eigenvalues of A (or s*E - A) lie inside the open unit disk,
%   otherwise a stabilizing initial feedback K0 is given as parameter.
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
%   | Info            | {0, 1}, used to disable/enable display of verbose |
%   |                 | status information during iteration steps         |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | K0              | matrix with dimensions m x n, used to stabilize   |
%   |                 | the spectrum of s*E - A, such that s*E - (A - BK0)|
%   |                 | has only d-stable eigenvalues                     |
%   |                 | (default zeros(m, n))                             |
%   +-----------------+---------------------------------------------------+
%   | dlyapopts       | structure, containing the optional parameters for |
%   |                 | Lyapunov equation solver, see ml_dlyap_smith_fac  |
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
%   Z    - full-rank solution factor of (1) or (2), such that X = Z*Z'
%   info - structure, containing the following information:
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsErr          | vector, containing the absolute error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%   | infoDLYAP       | array of structs, containing information about    |
%   |                 | the used Lyapunov equations solver for every      |
%   |                 | iteration step, see ml_dlyap_smith_fac            |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | RelErr          | vector, containing the relative error of the      |
%   |                 | solution matrix in each iteration step            |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
%   G. A. Hewer, An iterative technique for the computation of steady state
%   gains for the discrete optimal regulator, IEEE TRans. Autom. Control,
%   vol. 16, pp. 382--384, 1971. https://doi.org/10.1109/TAC.1971.1099755
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

narginchk(3, 5);

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

if ml_field_set_to_value(opts, 'Info')
    ml_assert_boolean(opts.Info, 'opts.Info');
else
    opts.Info = false;
end

if ml_field_set_to_value(opts, 'dlyapopts')
    assert(isa(opts.dlyapopts, 'struct'), ...
        'MORLAB:data', ...
        'The parameter opts.dlyapopts has to be a struct!');
else
    opts.dlyapopts = struct();
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

if ml_field_set_to_value(opts, 'K0')
    assert(isa(opts.K0, 'double') && isequal(size(opts.K0), [m n]), ...
        'MORLAB:data', ...
        'The stabilization matrix K0 must have the dimensions %d x %d!',...
        m, n);

    K = opts.K0;
else
    K = zeros(m, n);
end

% Case of empty data.
if isempty(A)
    Z     = [];
    info  = struct([]);
    return;
end

% Initial info structure.
info = struct();


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Im = eye(m);

niter            = 1;
converged        = 0;
[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEWTON-KLEINMAN ITERATION.                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    W              = [C; K];
    [Z, infoDLYAP] = ml_dlyap_smith_fac((A - B * K)', W', E', ...
        opts.dlyapopts);

    BZ = B' * Z;
    ZA = Z' * A;
    if hasE
        EZ = E' * Z;
    else
        EZ = Z;
    end
    K  = ((BZ * BZ' + Im) \ BZ) * ZA;

    % Information about current iteration step.
    abserr(niter) = norm(ZA' * (ZA - (Z' * B)*K) + C'*C - EZ * EZ', 'fro');
    relerr(niter) = abserr(niter) / max(norm(Z' * Z, 'fro'), 1);

    if opts.Info
        fprintf(1, ['DARE_NWT_FAC step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Save Lyapunov solver information.
    if nargout == 2
        info.infoDLYAP(niter) = infoDLYAP;
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
