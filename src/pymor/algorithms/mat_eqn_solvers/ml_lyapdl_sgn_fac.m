function [R, L, info] = ml_lyapdl_sgn_fac(A, B, C, E, opts)
%ML_LYAPDL_SGN_FAC Continuous-time dual Lyapunov equation solver.
%
% SYNTAX:
%   [R, L, info] = ML_LYAPDL_SGN_FAC(A, B, C)
%   [R, L, info] = ML_LYAPDL_SGN_FAC(A, B, C, [])
%   [R, L, info] = ML_LYAPDL_SGN_FAC(A, B, C, [], opts)
%
%   [R, L, info] = ML_LYAPDL_SGN_FAC(A, B, C, E)
%   [R, L, info] = ML_LYAPDL_SGN_FAC(A, B, C, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solutions of the dual standard Lyapunov
%   equations
%
%       A*X  + X*A' + B*B' = 0,                                         (1)
%       A'*Y + Y*A  + C'*C = 0,                                         (2)
%
%   or of the dual generalized Lyapunov equations
%
%       A*X*E' + E*X*A' + B*B' = 0,                                     (3)
%       A'*Y*E + E'*Y*A + C'*C = 0,                                     (4)
%
%   with X = R*R' and Y = L*L', using the sign function iteration. It is
%   assumed that the eigenvalues of A (or s*E - A) lie in the open left
%   half-plane.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1), (2) or (3), (4)
%   B    - matrix with dimensions n x m in (1) or (3)
%   C    - matrix with dimensions p x n in (2) or (4)
%   E    - matrix with dimensions n x n in (3), (4),
%          if empty the standard equations (1), (2) are solved
%   opts - structure, containing the following optional entries:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | AbsTol          | nonnegative scalar, tolerance for the absolute    |
%   |                 | error in the last iteration step                  |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | CompTol         | nonnegative scalar, tolerance for the column and  |
%   |                 | row compression during the iteration              |
%   |                 | (default 1.0e-02*sqrt(n*eps))                     |
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
%   R    - full-rank solution factor of (1), such that X = R*R'
%   L    - full-rank solution factor of (2), such that Y = L*L'
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
%   P. Benner, J. M. Claver, E. S. Quintana-Orti, Efficient solution of
%   coupled Lyapunov equations via matrix sign function iteration, in:
%   Proc. 3rd Portuguese Conf. on Automatic Control CONTROLO'98, Coimbra,
%   1998, pp. 205--210.
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
    opts.CompTol = 1.0e-02 * sqrt(n * eps);
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
    [R, L] = deal([]);
    info   = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R         = B;
niter     = 1;
converged = 0;

if hasE
    L     = C / E;
    normE = norm(E, 'fro');
else
    L     = C;
    normE = sqrt(n);
end

[abserr, relerr] = deal(zeros(1, opts.MaxIter));


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DUAL SIGN FUNCTION ITERATION.                                           %
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

    % Contruction of next R with column compression.
    R = sqrt(c1) * ml_compress_fac([R, c * EAinv * R], ...
        opts.CompTol, 'col');

    % Construction of next L with row compression.
    L = sqrt(c1) * ml_compress_fac([L; c * L * EAinv], ...
        opts.CompTol, 'row');

    % Construction of next A matrix.
    A = c1 * A + (0.5 * c) * EAinvE;

    % Information about current iteration step.
    abserr(niter) = norm(A + E, 'fro');
    relerr(niter) = abserr(niter) / normE;

    if opts.Info
        fprintf(1, ['LYAPDL_SGN_FAC step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if hasE
    R = sqrt(0.5) * (E \ R);
else
    R = sqrt(0.5) * R;
end
L = sqrt(0.5) * L';

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
