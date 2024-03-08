# -*- coding: utf-8 -*-
"""
Implements the weghting of low modulation rates in Section 7.1.5.4 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

from mosqito.sq_metrics.roughness.roughness_ecma._f_max import _f_max
from mosqito.sq_metrics.roughness.roughness_ecma._weight_factor_G import _weight_factor_G


def _weight_low_mod_rates(f_p_imax, A_hat, F_z):
    """
    Implements the weighting of low modulation rates in Eqs. 95 to 96 of 
    Section 7.1.5.2 of ECMA-418-2 (2nd Ed, 2022) standard for calculating
    Roughness.
    
    Parameters
    ----------
    f_p_imax : float
        Estimated fundamental modulation rate of the envelope.
    
    A_hat : (N_peaks,)-shaped numpy.array
        Weighted peaks' amplitudes.
    
    F_z : int
        Centre frequency of the current critical frequency band.
    
    Returns
    -------
    A : float
        Summed and weighted amplitudes of modulation rates' peaks.
    """

    # Coefficients 'q1' and 'q2' for weighting factor 'G'
    q1_low = 0.7066
    
    # Eq. 96
    q2_low = 1.0967 - 0.0640 * np.log2(F_z/1000.)
        
    # 'f_max' is the modulation rate at which the weighting factor G
    # reaches a maximum of 1 (Eq. 86)
    f_max = _f_max(F_z)
    
    # weighting factor G (Eq. 85)
    # G = _weight_factor_G(f_pi_hat, f_max, q1_low, q2_low)
    G = _weight_factor_G(f_p_imax, f_max, q1_low, q2_low)
    
    # Summation and weighting (Eq. 95)
    if f_p_imax < f_max:
        A = np.sum(G * A_hat)
    else:
        A = np.sum(A_hat)

    # Values of A below a threshold are set to zero
    if A < 0.074376:
        A = 0.

    return A
            