# -*- coding: utf-8 -*-
"""
Implements the error correction function on Table 10, Section 7.1.5.1 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

def _error_correction():
    """
    Calculates the error correction 'E(theta)' function, as defined in Table 10
    of the ECMA-418-2 (2nd Ed, 2022) standard, and used for calculating Roughness.
    
    Returns
    -------
    theta : numpy.array
        Array of 'theta' values, from 0 to 33
        
    E : float
        Array of error correction 'E(theta)' values
    """
    
    theta = np.arange(34)
    
    E = np.array([0.0000,   0.0457,     0.0907,     0.1346,     0.1765,     0.2157,     0.2515,     0.2828,     0.3084,
                  0.3269,   0.3364,     0.3348,     0.3188,     0.2844,     0.2259,     0.1351,     0.0000,     -0.1351,
                  -0.2259,  -0.2844,    -0.3188,    -0.3348,    -0.3364,    -0.3269,    -0.3084,    -0.2828,    -0.2515,
                  -0.2157,  -0.1765,    -0.1346,    -0.0907,    -0.0457,    0.0000,     0.0000])
    
    return theta, E