# -*- coding: utf-8 -*-
"""
Implements the 'beta' function on Eq. 79, Section 7.1.5.1 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

from mosqito.sq_metrics.roughness.roughness_ecma._error_correction import _error_correction

def _beta(theta, f_tilde, delta_f, E):
    """
    Calculates 'beta(theta)', the theoretical error after applying a correction
    factor function, as defined in Eq. 79 of the ECMA-418-2 (2nd Ed, 2022)
    standard, and used for calculating Roughness.
    
    Parameters
    ----------
    theta : float
        Input parameter
    
    Returns
    -------
    beta : float
        Output parameter
    """
    
    beta = ((np.floor(f_tilde/delta_f) + theta/32) * delta_f
            - (f_tilde + E))
    
    return beta