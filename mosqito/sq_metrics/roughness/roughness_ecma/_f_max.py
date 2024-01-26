# -*- coding: utf-8 -*-
"""
Implements Eq. 86 of Section 7.1.5.2 of ECMA-418-2 (2nd Ed, 2022) standard
for calculating Roughness. 

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

def _f_max(F_z):
    """
    Implements Eq. 86 of Section 7.1.5.2 of ECMA-418-2 (2nd Ed, 2022) standard
    for calculating Roughness. 
    
    Parameters
    ----------
    F_z : int
        Centre frequency of the current critical frequency band.
    
    Returns
    -------
    f_max : float
        Modulation rate at which the weighting factor G (Eq. 85) reaches a maximum of 1 
    """
    # Eq. 86
    f_max = 72.6937 * (1. - 1.1739*np.exp( -5.4583 * F_z / 1000. ))
    
    return f_max
            