# -*- coding: utf-8 -*-
"""
Implements Von Hann windowing function based on ECMA-418-2 (2nd Ed, 2022)
standard, used for calculating Roughness.

Uses the definition given Section 7.1.3, footnote [28].

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np


def _von_hann():
    """
    Calculates a specific version of the Von Hann window, as defined in 
    the ECMA-418-2 (2nd Ed, 2022) standard, and used for calculating Roughness.
    
    Returns
    -------
    hann : (512,)-shaped numpy.array
        Array containing the samples of the Von Hann weighting function.
    """
    
    N = 512
    
    arg1 = 2*np.pi*np.arange(N)/512
    
    return (0.5 - 0.5*np.cos(arg1)) / np.sqrt(0.375)