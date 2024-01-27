# -*- coding: utf-8 -*-
"""
Implements the weghting of high modulation rates in Section 7.1.5.2 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

from mosqito.sq_metrics.roughness.roughness_ecma._f_max import _f_max
from mosqito.sq_metrics.roughness.roughness_ecma._weight_factor_G import _weight_factor_G


def _weight_high_mod_rates(f_pi, A_pi, F_z):
    """
    Implements the weighting of high modulation rates in Eqs. 83 to 86 of 
    Section 7.1.5.2 of ECMA-418-2 (2nd Ed, 2022) standard for calculating
    Roughness. It uses a modulation-dependent factor and a scaling factor to
    adjust the calculations to the results of listening tests.
    
    Parameters
    ----------
    f_pi : (N_peaks,)-shaped numpy.array
        Estimated modulation frequencies of maxima in 'Phi_hat_z_l'
    
    A_pi : (N_peaks)-shaped numpy.array
        Estimated amplitudes of maxima in 'Phi_hat_z_l'
    
    F_z : int
        Centre frequency of the current critical frequency band.
    
    Returns
    -------
    A_pi_tilde : (N_peaks,)-shaped numpy.array
        Weighted amplitudes of 'N_peaks' maxima in 'Phi_hat_z_l'
    """

    
    q1 = 1.2822
    
    # Eq. 87
    if F_z/1000. < 2**(-3.4253):
        q2 = 0.2471
    else:
        q2 = 0.2471 + 0.0129 * (np.log2(F_z/1000.) + 3.4253)**2
        
    # 'f_max' is the modulation rate at which the weighting factor G
    # reaches a maximum of 1 (Eq. 86)
    f_max = _f_max(F_z)
                     
    # Parameters for r_max (Table 11)
    if F_z < 1000.:
        r1 = 0.3560
        r2 = 0.8049
    else:
        r1 = 0.8024
        r2 = 0.9333
    
    # scaling factor r_max (Eq. 84)
    r_max = 1. / (1. + r1 * np.abs( np.log2(F_z/1000.) )**r2 )
    
    # weighting factor G (Eq. 85)
    G = _weight_factor_G(f_pi, f_max, q1, q2)
    
    # Weighted peaks' amplitudes (Eq. 83)
    A_pi_tilde = A_pi * r_max
    A_pi_tilde[ f_pi >= f_max ] *= G[f_pi >= f_max]
            
    return A_pi_tilde
            
