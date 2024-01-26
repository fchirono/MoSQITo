# -*- coding: utf-8 -*-
"""
Implements the envelope noise reduction functions in Section 7.1.4 of
ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sim

def _env_noise_reduction(Phi_env):
    """
    Implements the envelope noise reduction functions in Section 7.1.4 of
    ECMA-418-2 (2nd Ed, 2022) standard, used for calculating Roughness. By
    default, 'sb_' = 512.
    
    
    Parameters
    ----------
    Phi_env : (53, L, sb_)-shaped numpy.array
        Array containing the scaled, downsampled power spectrum for 53 critical
        bands, 'L' time steps, and 'sb_' frequency samples.
    
    
    Returns
    -------
    Phi_hat : (53, L, sb_)-shaped numpy.array
        Array containing the noise-reduced power spectrum.
    """
    
    _, L, sb_ = Phi_env.shape
    
    if sb_ != 512:
        print(f"WARNING: sb_={sb_} is not 512, as required by the ECMA-418-2 standard!")
    
    # Noise reduction of the envelopes
    
    # averaging across neighboring critical bands
    two_dim_filter = np.array([1/3, 1/3, 1/3])
    
    Phi_avg = sim.convolve(Phi_env,
                           two_dim_filter[:, np.newaxis, np.newaxis],
                           mode='constant',
                           cval=0.)
    
    # ------------------------------------------------------------------------
    # plot averaged power spectrum for one time segment
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # plt.figure()
    # plt.pcolormesh(f, bark_axis,
    #                 10*np.log10(Phi_avg[:, timestep_to_plot, :sb_//2+1]))
    # plt.title(f'Averaged power spectrum of envelopes')
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # ------------------------------------------------------------------------
    
    # Sum the averaged Power spectra to get an overview of all the modulation
    # patterns over time (Eq. 68)
    s_ = np.sum(Phi_avg, axis=0)
    
    # median of 's_(L, k)' between k=2 and k=255
    k_range = np.arange(2, sb_//2)
    s_tilde = np.median(s_[:, k_range], axis=-1)

    # 's_tilde' becomes small compared to the peaks for modulated signals,
    # whereas 's_tilde' and the peaks have comparable amplitude for unmodulated
    # signals
    
    delta = 1e-10
    
    # k = 0, ..., 511
    k = np.arange(sb_)
    
    # Eq. 71
    #   --> 's_ / s_tilde' have large ratios for modulated signals
    w_tilde = (0.0856
               * (s_ / (s_tilde[:, np.newaxis] + delta))
               * np.clip(0.1891 * np.exp(0.0120 * k), 0., 1.) )
    
    # Eq. 70
    w_tilde_max = np.max(w_tilde[:, k_range], axis=-1)
    w_mask = (w_tilde >= 0.05 * w_tilde_max[:, np.newaxis])
    
    w = np.zeros(s_.shape)
    w[w_mask] = np.clip( w_tilde[w_mask] - 0.1407, 0., 1.)
    
    # 'w' tends to 0 for unmodulated signals, and to 1 for modulated signals
    # --> For white gaussian noise of 80 dB, all weights 'w' become 0 and
    #   result in a roughness of 0 Asper
    
    # ------------------------------------------------------------------------
    # plot weighting value 'w' for one time segment
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # plt.figure()
    # plt.pcolormesh(f, time_array[0, :], w[:, :sb_//2+1])
    # plt.title(f"Weighting coefficient 'w' (Eq. 70)")
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Time step [s]')
    # plt.colorbar()
    # ------------------------------------------------------------------------
    
    # weighted, averaged Power Spectra (Eq. 69)
    Phi_hat = Phi_avg*w
    
    return Phi_hat
