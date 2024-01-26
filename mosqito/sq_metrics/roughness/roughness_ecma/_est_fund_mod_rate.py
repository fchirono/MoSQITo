# -*- coding: utf-8 -*-
"""
Implements the estimation of the fundamental modulation rate in Section 7.1.5.3
of ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

def _est_fund_mod_rate(f_pi, A_pi_tilde):
    """
    Estimates the fundamental modulation rate from an array of power spectrum 
    peaks at the current critical band and time step.
    
    Parameters
    ----------
    f_pi : (N_peaks,)-shaped numpy.array
        Estimated modulation frequencies of maxima in 'Phi_hat_z_l'
    
    A_pi_tilde : (N_peaks,)-shaped numpy.array
        Weighted amplitudes of 'N_peaks' maxima in 'Phi_hat_z_l'
    
    Returns
    -------

    """
    
    # # dummy data
    # f_pi = np.arange(3, 9)
    # A_pi_tilde = np.random.random(6)
    
    E_i0 = np.zeros(f_pi.shape[0])
    
    I_all = []
    
    # for each candidate peak 'i0'...
    for i0 in range(f_pi.shape[0]):
        
        I_i0_candidate = []

        # integer ratios of all peaks' modulation rates to the current
        # peak modulation rate (Eq. 88)
        R_i0 = np.round(f_pi / f_pi[i0])
        
        # check ratios for repeated integer values
        values, counts = np.unique(R_i0, return_counts=True)
        
        # add unique elements to list of candidates
        for c in values[counts==1]:
            ic = np.nonzero(R_i0 == c)[0]
            I_i0_candidate.append( ic[0] )
        
        # iterate over repeated values and pick the one that minimizes 'crit'
        # (Eq. 89)
        for c in values[counts>1]:
            ic = np.nonzero(R_i0 == c)[0]
            
            crit = np.abs( (f_pi[ic] / (R_i0[ic] * f_pi[i0])) - 1)
        
            I_i0_candidate.append( ic[np.argmin(crit)] )
        
        # Create set 'I_i0' of indices of all maxima that belong to a harmonic
        # complex with fundamental modulation rate f_pi[i0], with a 4%
        # tolerance (Eq. 90)
        I_i0 = [i for i in I_i0_candidate
                if np.abs((f_pi[i] / (R_i0[i] * f_pi[i0])) - 1) < 0.04 ]
        
        I_all.append(I_i0)
        
        # Energy of the harmonic complex (Eq. 91)
        E_i0[i0] = np.sum(A_pi_tilde[I_i0])
        
    # find index 'i' that maximizes energy, and get the set 'I_i0'
    # corresponding to that index
    i_max = np.argmax(E_i0)
    I_max = I_all[i_max]
    
    # Fundamental modulation rate of the envelope
    f_p_imax = f_pi[i_max]
    
    # -------------------------------------------------------------------------
    # WARNING: list of indices 'I_max' is not sorted (i.e. from low to high)!
    # All variables 'X' are referred to as 'X[I_max]' from here onwards to 
    # account for 'I_max' ordering!
    # -------------------------------------------------------------------------
    
    # Peak amplitudes' weighting 
    
    # (Eq. 94)
    i_peak = np.argmax( A_pi_tilde[I_max] )
    
    # "center of gravity" (C.G.) of the peaks
    cg_peaks = (np.sum( f_pi[I_max] * A_pi_tilde[I_max] )
                / np.sum( A_pi_tilde[I_max] ))
    
    # weight factor based on distance of peaks' C.G. and peak with highest
    # amplitude (Eq. 93)
    w_peak = 1. + 0.1 * np.abs( cg_peaks - (f_pi[I_max])[i_peak] )**0.749
    
    A_hat = w_peak * A_pi_tilde[I_max]
    
    f_pi_hat = f_pi[I_max]
    
    return f_p_imax, f_pi_hat, A_hat