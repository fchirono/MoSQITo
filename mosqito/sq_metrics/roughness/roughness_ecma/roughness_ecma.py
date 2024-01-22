# -*- coding: utf-8 -*-
"""
Base function for Roughness calculation, according to ECMA-418-2, 2nd Ed (2022)

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""


# Standard imports
import numpy as np


# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import (
    _band_pass_signals,
)
from mosqito.sq_metrics.loudness.loudness_ecma._nonlinearity import _nonlinearity
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import _ecma_time_segmentation


# Data import
# Threshold in quiet
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z


def roughness_ecma(signal, fs, sb=2048, sh=1024):
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
    
    # ************************************************************************
    # Section 5.1.6 of ECMA-418-2, 2nd Ed. (2022)
    # Rectification (Eq. 21)
    
    block_array_rect = np.clip(block_array, a_min=0.00, a_max=None)

    # ************************************************************************
    # Sections 5.1.7 to 5.1.9 of ECMA-418-2, 2nd Ed. (2022)
    
    FS_specific = []
    for band_number in range(53):
        # ROOT-MEAN-SQUARE (section 5.1.7)
        # After the segmentation of the signal into blocks, root-mean square values of each block are calculated
        # according to Formula 17.
        rms_block_value = np.sqrt(
            2 * np.mean(np.array(block_array_rect[band_number]) ** 2, axis=1)
        )

        # NON-LINEARITY (section 5.1.8)
        # This section covers the other part of the calculations needed to consider the non-linear transformation
        # of sound pressure to specific loudness that does the the auditory system. After this point, the
        # computation is done equally to every block in which we have divided our signal.
        a_prime = _nonlinearity(rms_block_value)

        # SPECIFIC LOUDNESS CONSIDERING THE THRESHOLD IN QUIET (section 5.1.9)
        # The next calculation helps us obtain the result for the specific loudness - specific loudness with
        # consideration of the lower threshold of hearing.
        a_prime[a_prime < ltq_z[band_number]] = ltq_z[band_number]
        N_prime = a_prime - ltq_z[band_number]
        FS_specific.append(N_prime)

    bark_axis = np.linspace(0.5, 26.5, num=53, endpoint=True)
    
    return FS_specific, bark_axis
