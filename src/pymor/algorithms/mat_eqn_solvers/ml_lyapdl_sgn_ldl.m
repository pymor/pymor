function [RZ, RY, LZ, LY, info] = ml_lyapdl_sgn_ldl(A, B, R, C, Q, E, opts)
%ML_LYAPDL_SGN_LDL Continuous-time dual Lyapunov equation solver.
%
% SYNTAX:
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, [], C, [])
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, R, C, Q)
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, R, C, Q, [])
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, R, C, Q, [], opts)
%
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, [], C, [], E)
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, R, C, Q, E)
%   [RZ, RY, LZ, LY, info] = ML_LYAPDL_SGN_LDL(A, B, R, C, Q, E, opts)
%
% DESCRIPTION:
%   Computes the full-rank solutions of the dual standard Lyapunov
%   equations
%
%       A*X1  + X1*A' + B*R*B' = 0,                                     (1)
%       A'*X2 + X2*A  + C'*Q*C = 0,                                     (2)
%
%   or of the dual generalized Lyapunov equations
%
%       A*X1*E' + E*X1*A' + B*R*B' = 0,                                 (3)
%       A'*X2*E + E'*X2*A + C'*Q*C = 0,                                 (4)
%
%   with X1 = RZ*RY*RZ' and X2 = LZ*LY*LZ', using the sign function
%   iteration. It is assumed that the eigenvalues of A (or s*E - A) lie in
%   the open left half-plane.
%
% INPUTS:
%   A    - matrix with dimensions n x n in (1), (2) or (3), (4)
%   B    - matrix with dimensions n x m in (1) or (3)
%   R    - symmetric matrix with dimensions m x m in (1) or (3),
%          if empty R is assumed to be the identity
%   C    - matrix with dimensions p x n in (2) or (4)
%   Q    - symmetric matrix with dimensions p x p in (2) or (4),
%          if empty Q is assumed to be the identity
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
%   RZ   - full-rank solution factor of (1) or (3), s.t. X1 = RZ*RY*RZ'
%   RY   - full-rank solution factor of (1) or (3), s.t. X1 = RZ*RY*RZ'
%   LZ   - full-rank solution factor of (2) or (4), s.t. X2 = LZ*LY*LZ'
%   LY   - full-rank solution factor of (2) or (4), s.t. X2 = LZ*LY*LZ'
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
% See also ml_lyapdl_sgn_fac, ml_lyap_sgn_ldl.

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

narginchk(5, 7);

if (nargin < 7) || isempty(opts)
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

if not(isempty(R))
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

if not(isempty(Q))
    assert(isequal(size(Q), [p, p]), ...
        'MORLAB:data', ...
        'The matrix Q must be square with the number of rows as C!');

    assert(norm(Q - Q', 'fro') < 1.0e-14, ...
        'MORLAB:data', ...
        'The matrix Q must be symmetric!');

    if issparse(Q), Q = full(Q); end
else
    Q = eye(p);
end

if (nargin >= 6) && not(isempty(E))
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
    [RZ, RY, LZ, LY] = deal([]);
    info             = struct([]);
    return;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RZ        = B;
RY        = R;
LY        = Q;
niter     = 1;
converged = 0;

[abserr, relerr] = deal(zeros(1, opts.MaxIter));

if hasE
    LZ    = C / E;
    normE = norm(E, 'fro');
else
    LZ    = C;
    normE = sqrt(n);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DUAL SIGN FUNCTION ITERATION.                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (niter <= opts.MaxIter) && not(converged)
    EAinv  = E / A;
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

    % Construction of next RZ and RY via LDL column compression.
    [RZ, RY] = ml_compress_ldl([RZ, EAinv * RZ], ...
        blkdiag(c1 * RY, c2 * RY), opts.CompTol, 'col');

    % Construction of next LZ and LY via LDL row compression.
    [LZ, LY] = ml_compress_ldl([LZ; LZ * EAinv], ...
        blkdiag(c1 * LY, c2 * LY), opts.CompTol, 'row');

    % Construction of next A matrix.
    A = c1 * A + c2 * EAinvE;

    % Information about current iteration step.
    abserr(niter) = norm(A + E, 'fro');
    relerr(niter) = abserr(niter) / normE;

    if opts.Info
        fprintf(1, ['LYAPDL_SGN_LDL step: %4d absolute error: %e' ...
            ' relative error: %e \n'], ...
            niter, abserr(niter), relerr(niter));
    end

    % Method is converged if absolute or relative errors are small enough.
    converged = (abserr(niter) <= opts.AbsTol) || ...
        (relerr(niter) <= opts.RelTol);
    niter     = niter + 1;
end

if hasE
    RZ = sqrt(0.5) * (E \ RZ);
else
    RZ = sqrt(0.5) * RZ;
end
LZ = sqrt(0.5) * LZ';

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
