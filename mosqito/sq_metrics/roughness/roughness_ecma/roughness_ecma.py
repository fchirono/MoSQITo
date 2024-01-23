# -*- coding: utf-8 -*-
"""
Base function for Roughness calculation, according to ECMA-418-2, 2nd Ed (2022)

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""


# Standard imports
import numpy as np

import scipy.signal as ssig

# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import (
    _band_pass_signals,
)
from mosqito.sq_metrics.loudness.loudness_ecma._nonlinearity import _nonlinearity
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import _ecma_time_segmentation


# Data import
# Threshold in quiet
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z

from mosqito.sq_metrics.roughness.roughness_ecma._von_hann import _von_hann

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
    
    # Envelope calculation using Hilbert Transform
    analytic_signal = ssig.hilbert(block_array)
    p_env = np.abs(analytic_signal)             # Eq. 65   
    
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
    
    # ************************************************************************
    # Section 7.1.4 of ECMA-418-2, 2nd Ed. (2022)
    
    # Noise reduction of the envelopes
    
    # ************************************************************************
    # Section 7.1.5 of ECMA-418-2, 2nd Ed. (2022)
    
    # Spectral weighting
    
    # 7.1.5.1. Peak picking
    
    # 7.1.5.2. Weighting of high modulation rates
    
    # 7.1.5.3. Estimation of fundamental modulation rate
    
    # 7.1.5.4. Weighting of low modulation rates
    
    # ************************************************************************
    # Section 7.1.6 of ECMA-418-2, 2nd Ed. (2022)
    
    # Optional entropy weighting based on randomness of the modulation rate
    
    # ************************************************************************
    # Section 7.1.7 of ECMA-418-2, 2nd Ed. (2022)
    
    # Calcuation of time-dependent specific roughness
    
    # ************************************************************************
    # Calculate bark scale
    
    bark_axis = np.linspace(0.5, 26.5, num=53, endpoint=True)
    
    return bandpass_signals, bark_axis
