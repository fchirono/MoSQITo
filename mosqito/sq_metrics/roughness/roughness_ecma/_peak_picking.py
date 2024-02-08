# -*- coding: utf-8 -*-
"""
Implements the peak picking functions in Section 7.1.5.1 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np
import scipy.signal as ssig
import matplotlib.pyplot as plt

from mosqito.sq_metrics.roughness.roughness_ecma._error_correction import _error_correction


def _peak_picking(Phi_hat_z_l, fs_):
    """
    Implements the peak picking functions in Section 7.1.5.1 of ECMA-418-2
    (2nd Ed, 2022) standard for calculating Roughness.
    
    This function takes an array of noise-reduced, downsampled power spectrum
    values for one critical frequency 'z' and one time step 'l', finds
    'N_peaks' maxima that fulfill certain conditons, and estimates the
    frequencies and amplitudes of these maxima.
    
    Parameters
    ----------
    Phi_hat_z_l : (sb_)-shaped numpy.array
        Array containing the noise-reduced power spectrum for one critical
        band 'z' and one time step 'l'.
    
    fs_ : float
        Sampling frequency of downsampled envelope signal
    
    Returns
    -------
    f_pi : (N_peaks,)-shaped numpy.array
        Estimated modulation frequencies for all maxima in 'Phi_hat_z_l'
    
    A_pi : (N_peaks,)-shaped numpy.array
        Estimated amplitudes for all maxima in 'Phi_hat_z_l'
    """
    
    # get number of points in power spectrum
    sb_ = Phi_hat_z_l.shape[0]
    
    # get range of indices to search over (k=2...255)
    k_range = np.arange(2, sb_//2)
    
    # find peaks, calculate their prominence (Scipy definition of
    # prominence matches the definition in ECMA-418-2, 2022)
    phi_peaks, peaks_dict = ssig.find_peaks(Phi_hat_z_l[k_range],
                                            prominence=[None, None])
    
    # compensate for k_range starting at k=2
    phi_peaks += 2
    
    if phi_peaks.size == 0:
        
        # if no peaks were found, return empty arrays
        return np.array([]), np.array([])
    
    else:
        # .........................................................................
        # Plot Phi_hat
        
        # df_ = fs_/sb_
        # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
        
        # plt.figure()
        # plt.plot(f, Phi_hat_z_l[:sb_//2+1], label='Phi hat')
        # plt.plot(f[phi_peaks], Phi_hat_z_l[phi_peaks], 'r*', label='Peaks')
        # plt.ylim([-1, 10])
        # plt.legend()
        # plt.ylabel('Magnitude')
        # plt.xlabel('Freq [Hz]')
        # plt.tight_layout()
        
        # # plt.savefig('07_PhiHat_Peaks.png')
        # .........................................................................
        
        # sort peak indices based on their prominences, from smallest to
        # largest prominence, and pick (up to) 10 peaks with highest prominence
        sort_indices = np.argsort(peaks_dict['prominences'])[-10:]
        
        # get peak indices sorted by increasing prominence
        phi_peaks_sorted = phi_peaks[sort_indices]
        
        # Check if Phi[peak] > 0.05 * max(Phi[all_peaks]) (Eq. 72)
        peak_is_tall = (Phi_hat_z_l[phi_peaks_sorted]
                        > 0.05 * np.max( Phi_hat_z_l[phi_peaks_sorted] ) )
        
        # list of peaks in current critical freq, time step, that match criteria
        tall_peaks = phi_peaks_sorted[peak_is_tall]
        
        # for each tall peak, implement a quadratic fit to estimate modulation
        # rate and amplitude
        f_pi = np.zeros(tall_peaks.shape[0])
        A_pi = np.zeros(tall_peaks.shape[0])
        
        for i, pk in enumerate(tall_peaks):
            
            # solution vector (Eq. 74)
            phi_vec = np.array([Phi_hat_z_l[pk-1],
                                Phi_hat_z_l[pk],
                                Phi_hat_z_l[pk+1]])
            
            # modulation index matrix (Eq. 75)
            K = np.array([[(pk-1)**2,   (pk-1), 1],
                          [(pk  )**2,     (pk), 1],
                          [(pk+1)**2,   (pk+1), 1]])
            
            # find vector of unknowns
            C = np.linalg.solve(K, phi_vec)
            
            # first corrected modulation rate (Eq. 76)
            delta_f_ = fs_ / sb_
            f_tilde = -C[1] / ( 2 * C[0] ) * delta_f_
            
            # get values of theta, E(theta), beta(theta)
            theta, E = _error_correction()
            
            # Eq. 79
            beta = ( ( np.floor( f_tilde / delta_f_ ) + theta/32) * delta_f_
                    - (f_tilde + E))
            
            # ------------------------------------------------------------            
            # calculate correction factor following Eq. 78 to 81
            
            # Eq. 80, for 0 <= theta <= 32
            theta_min_i = np.argmin(np.abs(beta[:-1]))
            
            # Eq. 81
            cond1 = (theta[theta_min_i] > 0)
            cond2 = (beta[theta_min_i] * beta[theta_min_i-1] < 0)                
            
            if (cond1 and cond2):
                
                # tci: theta_corr_index
                tci = theta_min_i
                
            else:
                tci = theta_min_i + 1
            
            # ....................................................
            # # Eq. 78 as published - THIS IS WRONG!
            # rho_ = (E[tci]
            #         - ( (E[tci] - E[tci-1])*
            #           beta[tci-1] / (beta[tci] - beta[tci-1])))
            # ....................................................
            
            # Eq. 78 - corrected
            rho = (E[tci]
                    - ( (E[tci] - E[tci-1])*
                      beta[tci] / (beta[tci] - beta[tci-1])))
            
            # .........................................................................
            # # plot figure comparing the different values for rho
            
            # # This entire approach (Eqs. 78 to 81) is identical to
            # # using linear interpolation with Numpy:
            # rho_np = np.interp(0, beta, E)
            
            # plt.figure()
            # plt.plot(beta, E, '^:', markersize=10, label='Table 10, Eq. 79')
            # plt.plot(0, rho_np, 'r*', markersize=15, label='np.interp')
            # plt.plot(0, rho_, 'bs', markersize=12, label='Eq. 78 (as published)')
            # plt.plot(0, rho, 'mo', markersize=8, label='Eq. 78 (corrected)')
            # plt.grid()
            # plt.xlabel(r'$\beta(\theta)$', fontsize=14)
            # plt.ylabel(r'E($\theta$)', fontsize=14)
            # plt.legend()
            # plt.xlim([-0.5, 0.2])
            # plt.ylim([-0.35, 0.])
            # plt.tight_layout()
            
            # # plt.savefig('07_LinearInterpolation.png')
            
            # .........................................................................
            
            # Corrected modulation rate (Eq. 77)
            f_pi[i] = f_tilde + rho
            
            # Peak amplitudes (Eq. 82)
            pk_range = np.array([pk-1, pk, pk+1])
            A_pi[i] = np.sum(Phi_hat_z_l[pk_range])
            
        return f_pi, A_pi
                
            