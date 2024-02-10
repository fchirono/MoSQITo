# -*- coding: utf-8 -*-
"""
Implements the lowpass filtering in Section 7.1.7 of ECMA-418-2 (2nd Ed, 2022)
standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

def _lowpass_filtering(R_hat, fs_50):
    """
    Implements the lowpass filtering in Section 7.1.7 of ECMA-418-2 (2nd Ed,
    2022) standard for calculating Roughness. It uses different time constants
    for the rising and falling slopes, as the perception of sound events rises
    quickly with the beginning of the event but decays slowly with the event
    end.
    
    Parameters
    ----------
    R_hat : (53, Nt)-shaped numpy.array
        Array of transformed, calibrated values.
    
    fs_50 : float
        Sampling frequency of interpolated calculations. Must be 50 Hz.
    
    Returns
    -------
    R_spec : (53, Nt)-shaped numpy.array
        Array of time-dependent specific roughness.
    """
    
    # find indices where R_hat increases or remains the same over time (e.g.
    # rising slopes)
    R_rising = np.zeros(R_hat.shape).astype(bool)
    R_rising[:, 1:] = (np.diff(R_hat, axis=-1) >= 0)
    
    # Find indices where R_hat decreases over time (e.g. falling slopes)
    R_falling = np.logical_not(R_rising)
    
    # filtering time constants (Eq. 110)
    tau = np.zeros(R_hat.shape)
    tau[R_rising] = 0.0625
    tau[R_falling] = 0.5000
    
    # time-dependent specific roughness (Eq. 109)
    exp_ = np.exp(-1 / ( fs_50 * tau ) )
    
    R_spec = np.zeros(R_hat.shape)
    
    # l_50 = 0
    R_spec[:, 0] = R_hat[:, 0]
    
    # l_50 >= 1
    R_spec[:, 1:] = ( R_hat[:, 1:] * (1 - exp_[:, 1:])
                     + R_hat[:, 0:-1] * exp_[:, 1:] )
    
    
    # -------------------------------------------------------------------------
    # select a critical band to plot
    
    # import matplotlib.pyplot as plt
    
    # z = 25
    
    # plt.figure()
    # plt.plot(R_spec[z, :], '*--', label='R_spec')
    # plt.legend()
    # plt.xlabel('Time [s]')
    # plt.ylabel('Roughness [estimate]')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('08_LowpassFiltering.png')
    # -------------------------------------------------------------------------
    
    return R_spec
            