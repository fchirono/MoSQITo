# -*- coding: utf-8 -*-
"""
Implements the nonlinear transformation in Section 7.1.7 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

def _nonlinear_transform(R_est):
    """
    Implements the nonlinear transformation and calibration in Section 7.1.7 of
    ECMA-418-2 (2nd Ed, 2022) standard for calculating Roughness.
    
    Parameters
    ----------
    R_est : (53, Nt)-shaped numpy.array
        Array of interpolated, clipped, uncalibrated Roughness values.
    
    Returns
    -------
    R_hat : (53, Nt)-shaped numpy.array
        Array of transformed, calibrated Roughness values.
    """

    # squared mean (Eq. 107)
    R_tilde = np.sqrt( np.sum(R_est**2, axis=0) / R_est.shape[0] )
    
    # linear mean (Eq. 108)
    R_lin = np.mean(R_est, axis=0)
    
    # calibration factor c_R [asper / Bark_HMS]
    c_R = 0.0180909
    
    # Eq. 106
    B_l = R_tilde / R_lin
    B_l[R_lin == 0] = 0.
    
    # Eq. 105
    arg_tanh = 1.6407 * (B_l - 2.5804)
    E_l = 0.95555*( np.tanh(arg_tanh) + 1)*0.5 + 0.58449
    
    # time-dependent specific roughness estimate (Eq. 104)
    R_hat = c_R * (R_est ** E_l)
    
    return R_hat
            