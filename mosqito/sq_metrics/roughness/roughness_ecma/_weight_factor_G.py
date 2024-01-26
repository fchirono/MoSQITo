# -*- coding: utf-8 -*-
"""
Implements Eq. 85 of Section 7.1.5.2 of ECMA-418-2 (2nd Ed, 2022) standard
for calculating Roughness. 

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

def _weight_factor_G(f_pi, f_max, q1, q2):
    """
    Implements Eq. 85 of Section 7.1.5.2 of ECMA-418-2 (2nd Ed, 2022) standard
    for calculating Roughness. 
    
    Parameters
    ----------
    f_pi : (N_peaks,)-shaped numpy.array
        Estimated modulation frequencies of maxima in 'Phi_hat_z_l'
    
    f_max : float
        Modulation rate at which the weighting factor G reaches a maximum of 1
    
    q1 : float
        Parameter 'q1'.
    
    q2 : float 
        Parameter 'q2'.
    
    Returns
    -------
    G : (N_peaks,)-shaped numpy.array
        Weighting factor defined in Equation 85.
    """
    
    # weighting factor G (Eq. 85)
    f1 = f_pi / f_max
    f2 = f_max / f_pi
    
    arg1 = ( (f1 - f2) * q1)**2
    
    G = 1. / ( (1. + arg1)**q2 )
    
    return G
            