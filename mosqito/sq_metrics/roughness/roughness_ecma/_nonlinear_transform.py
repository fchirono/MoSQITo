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
    Rlin_nonzero = (R_lin != 0)
    B_l = np.zeros(R_tilde.shape)
    B_l[Rlin_nonzero] = R_tilde[Rlin_nonzero] / R_lin[Rlin_nonzero]
    
    # Eq. 105
    arg_tanh = 1.6407 * (B_l - 2.5804)
    E_l = 0.95555*( np.tanh(arg_tanh) + 1)*0.5 + 0.58449
    
    # time-dependent specific roughness estimate (Eq. 104)
    R_hat = c_R * (R_est ** E_l)
    
    # -------------------------------------------------------------------------
    # select a critical band to plot
    
    # import matplotlib.pyplot as plt
    # from mosqito.utils.conversion import bark2freq
    
    # bark_axis = np.linspace(0.5, 26.5, num=53, endpoint=True)
    
    # z = 35
    
    # plt.figure()
    # plt.plot(R_hat[z, :], '*--', label='R_hat')
    # plt.legend()
    # plt.xlabel('Time [s]')
    # plt.ylabel('Roughness [estimate]')
    # plt.grid()
    # plt.title(f'{bark_axis[z]:1.1f} Bark ({bark2freq(bark_axis[z]):1.0f} Hz)')
    # plt.tight_layout()
    # plt.savefig('08_NonlinearTransform.png')
    # -------------------------------------------------------------------------
    
    return R_hat
            