# -*- coding: utf-8 -*-
"""
Implements the interpolation to 50 Hz sampling frequency in Section 7.1.7 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

import scipy.interpolate as si


def _interpolate_to_50Hz(A, time_array, n_samples, fs):
    """
    Implements the interpolation to 50 Hz sampling frequency and clipping of 
    negative values in Section 7.1.7 of ECMA-418-2 (2nd Ed, 2022) standard for
    calculating Roughness.
    
    Parameters
    ----------
    A : (53, L)-shaped numpy.array
        Array of modulation amplitudes per critical band 'z' and time step 'l'
    
    time_array : (53, L)-shaped numpy.array
        Array of time steps per critical band 'z' and time step 'l'
    
    n_samples : int
        Number of samples in the original sound pressure signal
    
    fs : float
        Original sampling frequency, must be 48 kHz
    
    Returns
    -------
    R_est : (53, Nt)-shaped numpy.array
        Array of interpolated, clipped, uncalibrated Roughness values.
        
    t_50 : (Nt,)-shaped numpy.array
        New time array for interpolated time steps
    
    fs_50 : float
        New sampling frequency of 50 Hz
    """

    # New sampling rate of 50 Hz
    fs_50 = 50.
    
    # Last sample to be evaluated (Eq. 103) - removes the results belonging to
    # the zero-padding done at the start of the processing
    l50_end = int(np.ceil(n_samples*fs_50/fs))
    
    t_50 = np.arange(l50_end+1) * (1./fs_50)
    
    # interpolate using Piecewise cubic Hermitian Interpolating Polynomial (PCHIP)
    R_est = si.pchip_interpolate(time_array[0, :], A, t_50, axis=-1)
    
    # -------------------------------------------------------------------------
    # select a critical band to plot
    
    # import matplotlib.pyplot as plt
    
    # z = 15
    
    # plt.figure()
    # plt.plot(time_array[z, :], A[z, :], 'o:', label='A')
    # plt.plot(t_50, R_est[z, :], '*--', label='R_est')
    # plt.legend()
    # plt.vlines(t_50[l50_end], np.min(A), np.max(A), color='k', linestyle='-.')
    # plt.grid()
    # -------------------------------------------------------------------------
    
    # set negative values to zero
    R_est = np.clip(R_est, 0., None)

    return R_est, t_50, fs_50
            