from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla


def fast_vf(
    omega,
    H,
    order,
    poles_estimation_threshold=1e-1,
    model_error_threshold=1e-3,
    maxiter=5,
    enforce_stability=True,
    debug=False,
):
    """Fast Vector Fitting, real-valued implementation from [1].

    Parameters
    ----------
    omega
        Frequency samples, vector.
        This is angular frequency (omega = 2*pi*f).
    H
        response samples, 3D array.
        First dimension corresponds to system outputs.
        Second dimension to system inputs.
        Third dimension corresponds to frequency.
    order
        Desired order of the VF model.
    poles_estimation_threshold
        Threshold for the first convergence.
        Test to stop the iterative estimation of poles.
    model_error_threshold
        Threshold for the second convergence test
        when modeling error falls below this threshold,
        the model is returned.
    maxiter
        Maximum number of iterations.
        When reached the algorithm stops even if target accuracy was not
        reached.
    enforce_stability
        Flip model poles into the stable left half plane to enforce model
        stability.
    debug
        Print and plot extra information.
        Pause at each iteration.

    Results
    -------
    data
        Dictionary with fields:
        :pr:
            Real poles of the model, vector.
        :pc:
            Complex poles of the model, one per conjugate pair, vector.
        :R0:
            Constant term R0 of the model, 2D array.
        :Rr:
            Residues of real poles, 3D array.
            First and second dimension correspond to outputs and inputs.
            Third dimension to poles.
        :Rc:
            Residues of complex poles, 3D array.
            First and second dimension correspond to outputs and inputs.
            Third dimension to poles.
            One residue per pair.

    Licensing condition:
    you can freely use these codes (the "Software") subject to the conditions in
    the LICENSE file. Note that you must cite the following book chapter in the
    publications and product documentation arising from the use of this Software

    [1] P. Triverio, "Vector Fitting", in P. Benner, S. Grivet-Talocia, A.
    Quarteroni, G. Rozza, W. H. A. Schilders, L. M. Silveira (Eds.),
    "Handbook on Model Order Reduction", De Gruyter (to appear).

    Copyright 2019 Piero Triverio, www.modelics.org
    """
    assert isinstance(omega, np.ndarray)
    assert omega.ndim == 1

    assert isinstance(H, np.ndarray)
    assert H.ndim == 3

    assert len(omega) == H.shape[2]

    ## initializations
    tic = perf_counter()

    kbar = len(omega)  # number of samples
    qbar, mbar, _ = H.shape  # number of outputs and inputs
    nbar = order  # desired model order

    data = {}

    ## initial poles
    # number of real poles
    nr = nbar % 2  # no real poles if nbar is even, one if order is odd
    # number of complex conjugate pairs
    nc = (nbar - nr) // 2

    alpha = 0.01

    # real pole
    if nr == 1:  # nbar is odd
        pr = np.array([-alpha * omega.max()])
    else:
        pr = np.array([])

    # complex poles
    if nc == 1:
        pc = np.array([(-alpha + 1j) * omega.max() / 2])
    elif omega.min() == 0:
        pc = (-alpha + 1j) * omega.max() / nc * np.arange(1, nc + 1)
    else:
        pc = (-alpha + 1j) * (omega.min() + (omega.max() - omega.min()) / (nc - 1) * np.arange(nc))

    ## poles estimation
    # iterative process
    iter = 1
    while iter <= maxiter:
        print(f'Iteration {iter}')

        ## QR decompositions
        # Compute the Phir and Phic matrices
        Phir, Phic = compute_phi_matrices(omega, pr, pc)

        # compute Phi0 and Phi1
        # Phi0 = [ones(kbar,1),Phir, Phic]
        # Phi1 = [Phir, Phic]

        # preallocate matrix used in QR decompositions
        M = np.zeros((2 * kbar, 2 * nbar + 1))
        # preallocate matrix and right hand side for least squares problem (54)
        Alsq = np.zeros((nbar * qbar * mbar, nbar))
        blsq = np.zeros(nbar * qbar * mbar)

        # compute the first columns of M, which do not depend on q and m
        M[:kbar, 0] = 1
        M[:kbar, 1 : nr + 1] = Phir.real
        M[:kbar, nr + 1 : nbar + 1] = Phic.real
        M[kbar:, 1 : nr + 1] = Phir.imag
        M[kbar:, nr + 1 : nbar + 1] = Phic.imag

        irow = 0
        for q in range(qbar):  # loop over outputs
            for m in range(mbar):  # loop over inputs
                V_Hqm = H[q, m, :]
                V_Hqm_T = V_Hqm[:, np.newaxis]
                # compute the second part of M
                M[:kbar, nbar + 1 : nbar + nr + 1] = -(V_Hqm_T * Phir).real
                M[:kbar, nbar + nr + 1 :] = -(V_Hqm_T * Phic).real
                M[kbar:, nbar + 1 : nbar + nr + 1] = -(V_Hqm_T * Phir).imag
                M[kbar:, nbar + nr + 1 :] = -(V_Hqm_T * Phic).imag
                # QR decomposition (53)
                Qqm, Rqm = spla.qr(M, mode='economic')

                # assemble the coefficient matrix and right hand side of the
                # least squares problem for poles estimation
                Alsq[irow : irow + nbar, :] = Rqm[nbar + 1 :, nbar + 1 :]
                blsq[irow : irow + nbar] = (
                    Qqm[:kbar, nbar + 1 :].T @ V_Hqm.real + Qqm[kbar:, nbar + 1 :].T @ V_Hqm.imag
                )

        # least squares problem (54)
        cw = spla.lstsq(Alsq, blsq)[0]

        # evaluate the weighting function before we compute the new poles
        # estimate
        # this will be used by the first convergence test
        w = 1 + Phir @ cw[:nr] + Phic @ cw[nr:]

        # plot the magnitude of the weighting function
        if debug:
            fig, ax = plt.subplots()
            ax.plot(omega, np.abs(w))
            ax.set_xlabel(r'$\omega$')
            ax.set_ylabel(r'$|w(i \omega)|$')
            ax.set_title('Weighting function magnitude')
            ax.grid()

        # Compute the new poles estimate
        A = np.zeros((nbar, nbar))
        bw = np.ones(nbar)
        for ii in range(nr):
            A[ii, ii] = pr[ii]
            # bw[ii] = 1
        for ii in range(nc):
            A[nr + 2 * ii : nr + 2 * ii + 2, nr + 2 * ii : nr + 2 * ii + 2] = [
                [pc[ii].real, pc[ii].imag],
                [-pc[ii].imag, pc[ii].real],
            ]
            bw[nr + 2 * ii : nr + 2 * ii + 2] = [2, 0]
        p_new = spla.eigvals(A - np.outer(bw, cw))

        # plot the new poles estimate
        if debug:
            fig, ax = plt.subplots()
            ax.plot(p_new.real, p_new.imag, 'ro', markerfacecolor='none')
            ax.set_xlabel('Re')
            ax.set_ylabel('Im')
            ax.set_title(f'Poles estimate, iteration {iter}')
            ax.grid()

        # extract real poles
        ind_rp = np.abs(p_new.imag) <= 10 * np.finfo(np.float64).eps * np.abs(p_new)
        pr = p_new[ind_rp].real
        nr = len(pr)

        # extract complex conjugate pairs of poles
        # find only the poles with positive imaginary part
        ind_cp = p_new.imag >= 10 * np.finfo(np.float64).eps * np.abs(p_new)
        pc = p_new[ind_cp]
        nc = len(pc)

        ## stability/causality enforcement
        if enforce_stability:
            pr = -np.abs(pr)
            pc = -np.abs(pc.real) + 1j * pc.imag

        ## first convergence test

        w_minus_one = 1 / np.sqrt(kbar) * spla.norm(np.abs(w - 1))
        # do a tentative model fitting if either:
        # - the first convergence test is successful
        # - debug is enabled
        print('Convergence test (poles estimation): ', end='')
        print('passed' if w_minus_one <= poles_estimation_threshold else 'failed', end='')
        print(f' ({w_minus_one})')
        if w_minus_one <= poles_estimation_threshold or debug:
            ## tentative final fitting
            Phir, Phic = compute_phi_matrices(omega, pr, pc)

            # compute the matrix of the least squares problem
            Alsq = np.zeros((2 * kbar, nbar + 1))
            Alsq[:kbar, 0] = 1
            Alsq[:kbar, 1 : nr + 1] = Phir.real
            Alsq[:kbar, nr + 1 : nbar + 1] = Phic.real
            Alsq[kbar:, 1 : nr + 1] = Phir.imag
            Alsq[kbar:, nr + 1 : nbar + 1] = Phic.imag

            # store model coefficients in the output structure Model, in case it
            # will found accurate enough
            data['pr'] = pr.copy()
            data['pc'] = pc.copy()
            data['R0'] = np.zeros((qbar, mbar))
            data['Rr'] = np.zeros((qbar, mbar, nr))
            data['Rc'] = np.zeros((qbar, mbar, nc), dtype=np.complex128)

            # model-samples error
            err = 0
            for q in range(qbar):
                for m in range(mbar):
                    # right-hand side
                    V_Hqm = H[q, m, :]
                    blsq = np.concatenate([V_Hqm.real, V_Hqm.imag])
                    c_Hqm = spla.lstsq(Alsq, blsq)[0]
                    data['R0'][q, m] = c_Hqm[0]
                    data['Rr'][q, m, :] = c_Hqm[1 : nr + 1]
                    data['Rc'][q, m, :] = c_Hqm[nr + 1 :: 2] + 1j * c_Hqm[nr + 2 :: 2]

                    # Plot the given samples vs the model response for the
                    # (1,1) entry of the transfer function (if in debug mode)
                    if debug and q == 1 and m == 1:
                        # compute model response
                        Htemp = compute_model_response(
                            omega,
                            data['R0'][q, m],
                            data['Rr'][q, m, :],
                            data['Rc'][q, m, :],
                            data['pr'],
                            data['pc'],
                        )
                        fig, axs = plt.subplots(1, 2)
                        ax = ax[0]
                        ax.plot(omega, np.abs(H[q, m, :]), 'bx', label='Samples H_k')
                        ax.plot(omega, np.abs(Htemp[q, m, :]), 'r-.', label='Model')
                        ax.set_xlabel('Omega')
                        ax.set_ylabel('Magnitude')
                        ax.legend()
                        ax.grid()

                        ax = axs[1]
                        ax.plot(
                            omega,
                            np.angle(H[q, m, :], deg=True),
                            'bx',
                            label='Samples H_k',
                        )
                        ax.plot(
                            omega,
                            np.angle(Htemp[q, m, :], deg=True),
                            'r-.',
                            label='Model',
                        )
                        ax.set_xlabel('Omega')
                        ax.set_ylabel('Phase [deg]')
                        ax.legend()

                    err += spla.norm(Alsq @ c_Hqm - blsq) ** 2
            err = np.sqrt(err) / np.sqrt(qbar * mbar * kbar)

            if err <= model_error_threshold:
                print(f'Convergence test (model-samples error): passed ({err})')
                print('Model identification successful')
                toc = perf_counter()
                print(f'Modeling time: {toc - tic}s')
                return data
            else:
                print(f'Convergence test (model-samples error): failed ({err})')

        if debug:
            plt.show()

        iter += 1

    print(
        'Warning: could not reach the desired modeling error'
        ' within the allowed number of iterations'
    )
    toc = perf_counter()
    print(f'Modeling time: {toc - tic}s')
    return data


def compute_phi_matrices(omega, pr, pc):
    """Function to compute the frequency response of a Vector Fitting model [1].

    Parameters
    ----------
    omega
        Frequency samples, vector.
        This is angular frequency (omega = 2*pi*f).
    pr
        Real poles.
    pc
        Complex conjugate poles (only one pole per pair).

    Returns
    -------
    Phir
        Coefficient matrix, part associated to real poles.
    Phic
        Coefficient matrix, part associated to complex poles.

    Licensing condition:
    you can freely use these codes (the "Software") subject to the conditions in
    the LICENSE file. Note that you must cite the following book chapter in the
    publications and product documentation arising from the use of this Software

    [1] P. Triverio, "Vector Fitting", in P. Benner, S. Grivet-Talocia, A.
    Quarteroni, G. Rozza, W. H. A. Schilders, L. M. Silveira (Eds.),
    "Handbook on Model Order Reduction", De Gruyter (to appear).

    Copyright 2019 Piero Triverio, www.modelics.org
    """
    omegaT = omega[:, np.newaxis]
    Phir = 1 / (1j * omegaT - pr)
    Phic = np.zeros((len(omega), 2 * len(pc)), dtype=np.complex128)
    Phic[:, ::2] = 1 / (1j * omegaT - pc) + 1 / (1j * omegaT - pc.conj())
    Phic[:, 1::2] = 1j / (1j * omegaT - pc) - 1j / (1j * omegaT - pc.conj())
    return Phir, Phic


def compute_model_response(omega, R0, Rr, Rc, pr, pc):
    """Compute the frequency response of a Vector Fitting model [1].

    Parameters
    ----------
    omega
        Frequency samples, vector.
        This is angular frequency (omega = 2*pi*f).
    R0
        Constant coefficient.
    Rr
        Residues of real poles, 3D array.
        First dimension corresponds to system outputs.
        Second dimension to system inputs.
        Third dimension corresponds to the various poles.
    Rc
        Residues of complex conjugate pole pairs (only one per pair).
    pr
        Real poles, vector.
    pc
        Complex poles, vector.
        Only one per pair of conjugate poles.

    Returns
    -------
    H
        Model response samples, 3D array.
        First dimension corresponds to system outputs.
        Second dimension to system inputs.
        Third dimension corresponds to frequency.

    Licensing condition:
    you can freely use these codes (the "Software") subject to the conditions in
    the LICENSE file. Note that you must cite the following book chapter in the
    publications and product documentation arising from the use of this Software

    [1] P. Triverio, "Vector Fitting", in P. Benner, S. Grivet-Talocia, A.
    Quarteroni, G. Rozza, W. H. A. Schilders, L. M. Silveira (Eds.),
    "Handbook on Model Order Reduction", De Gruyter (to appear).

    Copyright 2019 Piero Triverio, www.modelics.org
    """
    qbar, mbar = R0.shape  # number of outputs and inputs
    kbar = len(omega)  # number of frequency points

    nr = len(pr)  # number of real poles
    nc = len(pc)  # number of complex conjugate pairs

    # preallocate space for H
    H = np.zeros((qbar, mbar, kbar), dtype=np.complex128)

    # this part should be vectorized for higher efficiency
    for ik in range(kbar):
        H[:, :, ik] = R0
        for ir in range(nr):
            H[:, :, ik] += Rr[:, :, ir] / (1j * omega[ik] - pr[ir])
        for ic in range(nc):
            H[:, :, ik] += Rc[:, :, ic] / (1j * omega[ik] - pc[ic])
            H[:, :, ik] += Rc[:, :, ic].conj() / (1j * omega[ik] - pc[ic].conj())

    return H
