# -*- coding: utf-8 -*-
"""
Base function for Roughness calculation, according to ECMA-418-2, 2nd Ed (2022)

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""


# Standard imports
import numpy as np

import matplotlib.pyplot as plt
import scipy.signal as ssig


# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma.loudness_ecma import loudness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import _band_pass_signals
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import _ecma_time_segmentation
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq

from mosqito.sq_metrics.roughness.roughness_ecma._von_hann import _von_hann
from mosqito.sq_metrics.roughness.roughness_ecma._env_noise_reduction import _env_noise_reduction
from mosqito.sq_metrics.roughness.roughness_ecma._peak_picking import _peak_picking
from mosqito.sq_metrics.roughness.roughness_ecma._weight_high_mod_rates import _weight_high_mod_rates
from mosqito.sq_metrics.roughness.roughness_ecma._est_fund_mod_rate import _est_fund_mod_rate

from mosqito.utils.conversion import bark2freq


def roughness_ecma(signal, fs, sb=16384, sh=4096):
    """[*WARNING*] Roughness calculation of a signal sampled at 48 kHz,
    according to ECMA-418-2, 2nd Ed (2022).

    *************************** WARNING! ************************************
    * The code is not finished yet, and does not return correct results!    *    
    *************************************************************************

    Parameters
    ----------
    signal :numpy.array  or DataTime object
        A time signal in Pa
        
    fs : float, optional
        Sampling frequency, in Hz.
        
    sb: int or list of int
        Block size, or list of block sizes per band
        
    sh: int or list of int
        Hop size, or list of hop sizes per band

    Returns
    -------
    R: list of numpy.array
        Time-dependent Specific Rughness [asper per Bark]. Each of the 53 element
        of the list corresponds to the time-dependent specific roughness for a
        given bark band. Can be a ragged array if a different sb/sh are used
        for each band.

    bark_axis: numpy.array
        Bark axis
    """
    
    assert fs == 48000, "Sampling frequency 'fs' must be 48 kHz!"
    
    n_samples = signal.shape[0]
    
    # ************************************************************************
    # Preliminary: calculate time-dependent specific loudness N'(l, z)
    
    # N_basis is (53, L)-shaped,, where L is the number of time segments
    N_basis, bark_axis = loudness_ecma(signal, sb, sh)
    N_basis = np.array(N_basis)
    
    L = N_basis.shape[1]
    
    F = _auditory_filters_centre_freq()
    
    # ************************************************************************
    # Section 5.1.2 of ECMA-418-2, 2nd Ed (2022)
    
    # -----------------------------------------------------------------------    
    # Apply windowing function to first 5 ms (240 samples)
    n_fadein = 240
    
    # Eq. (1)
    w_fadein = 0.5 - 0.5*np.cos(np.pi*np.arange(n_fadein)/n_fadein)
    
    signal[:240] *= w_fadein
    
    # -----------------------------------------------------------------------    
    # Calculate zero padding at start and end of signal
    sb_max = np.max(sb)
    sh_max = np.max(sh)
    
    n_zeros_start = sb_max
    
    # Eqs. (2), (3) 
    n_new = sh_max * (np.ceil((n_samples + sh_max + sb_max)/(sh_max)) - 1)
    
    n_zeros_end = int(n_new) - n_samples
    
    signal = np.concatenate( (np.zeros(n_zeros_start),
                              signal,
                              np.zeros(n_zeros_end)))
    
    # ************************************************************************
    # Sections 5.1.3 to 5.1.4 of ECMA-418-2, 2nd Ed. (2022)
    
    # Computaton of band-pass signals
    bandpass_signals = _band_pass_signals(signal)

    # ************************************************************************
    # Section 5.1.5 of ECMA-418-2, 2nd Ed. (2022)
    
    # segmentation into blocks
    block_array, time_array = _ecma_time_segmentation(bandpass_signals, sb, sh,
                                                      n_new)
    
    # block_array is (53, L, sb)-shaped, where L is number of time segments
    block_array = np.array(block_array)
    
    # time_array is (53, L)-shaped
    time_array = np.array(time_array)
    
    # ************************************************************************
    # Section 7.1.2 of ECMA-418-2, 2nd Ed. (2022)
    
    # Envelope calculation using Hilbert Transform (Eq. 65)
    analytic_signal = ssig.hilbert(block_array)
    p_env = np.abs(analytic_signal)
    
    # ------------------------------------------------------------------------
    # # plot envelope and bandpass signal for one segment
    
    # band_to_plot = 35
    # timestep_to_plot = 8
    
    # t = np.linspace(0, (sb-1)/fs, sb)
    
    # plt.figure()
    # plt.plot(t, p_env[band_to_plot, timestep_to_plot, :],
    #           label='Envelope')
    # plt.plot(t, block_array[band_to_plot, timestep_to_plot, :], ':',
    #           label='Bandpass Signal')
    # plt.legend()
    # plt.title(f'{band_to_plot} Bark ({bark2freq(band_to_plot)} Hz)')
    # ------------------------------------------------------------------------
    
    # Downsampling by a factor of 32
    downsampling_factor = 32
    p_env_downsampled_ = ssig.decimate(p_env, downsampling_factor//4)
    p_env_downsampled = ssig.decimate(p_env_downsampled_, 4)
    
    # new downsampled sampling freq, block size, hop size
    fs_ = fs//downsampling_factor       # 1500 Hz
    sb_ = sb//downsampling_factor       # 512 points
    sh_ = sh//downsampling_factor       # 128 points
    
    # ************************************************************************
    # Section 7.1.3 of ECMA-418-2, 2nd Ed. (2022)
    
    # Calculation of scaled power spectrum
    
    # get Von Hann window coefficients
    hann = _von_hann()
    
    # max Loudness (Eq. 67)
    N_max = np.max(N_basis, axis=0)
    Phi_env_zero = np.sum( (hann*p_env_downsampled)**2 , axis=-1)
    
    
    # Scaled power spectrum (Eq. 66)
    scaling = np.zeros(N_basis.shape)
    non_zero = (N_max*Phi_env_zero != 0)
    scaling[non_zero] = (N_basis**2)[non_zero] / (N_max * Phi_env_zero)[non_zero]
    
    #   'Phi_env' is (53, L, sb_)-shaped
    Phi_env = (scaling[:, :, np.newaxis]
               * np.abs(np.fft.fft(hann*p_env_downsampled))**2 )
    
    # ------------------------------------------------------------------------
    # plot scaled power spectrum for one time segment
    
    # timestep_to_plot = 8
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # plt.figure()
    # plt.pcolormesh(f, bark_axis,
    #                 10*np.log10(Phi_env[:, timestep_to_plot, :sb_//2+1]))
    # plt.title(f'Scaled power spectrum of envelopes')
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # ------------------------------------------------------------------------
    
    # ************************************************************************
    # Section 7.1.4 of ECMA-418-2, 2nd Ed. (2022)
    
    Phi_hat = _env_noise_reduction(Phi_env)
    
    # ************************************************************************
    # Section 7.1.5 of ECMA-418-2, 2nd Ed. (2022)
    
    # Spectral weighting
    
    # for each critical freq...
    for z in range(53):
        
        # for each time step...
        for l in range(L):
            
            # TODO: code might not find any peaks!
            
            # test values
            z = 20
            l = 8
            
            # ------------------------------------------------------------
            # 7.1.5.1. Peak picking
            f_pi, A_pi = _peak_picking(Phi_hat[z, l], fs_)
            
            # 7.1.5.2. Weighting of high modulation rates
            A_pi_tilde = _weight_high_mod_rates(f_pi, A_pi, F[z])
            
            # ------------------------------------------------------------
            # 7.1.5.3. Estimation of fundamental modulation rate
            f_pi_hat, A_hat = _est_fund_mod_rate(f_pi, A_pi_tilde)
            
            # ------------------------------------------------------------
            # 7.1.5.4. Weighting of low modulation rates
            
    
    # ************************************************************************
    # Section 7.1.6 of ECMA-418-2, 2nd Ed. (2022)
    
    # Optional entropy weighting based on randomness of the modulation rate
    
    # NOT IMPLEMENTED!
    
    # ************************************************************************
    # Section 7.1.7 of ECMA-418-2, 2nd Ed. (2022)
    
    # Calcuation of time-dependent specific roughness
    
    # ************************************************************************

    
    return bandpass_signals, bark_axis
