# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sp_signal

from mosqito.sq_metrics.loudness.loudness_ecma._ear_filter_design import (
    _ear_filter_design,
)
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import (
    _auditory_filters_centre_freq,
)
from mosqito.sq_metrics.loudness.loudness_ecma._gammatone import _gammatone
from mosqito.utils import time_segmentation


def _band_pass_signals(sig, sb=2048, sh=1024):
    """Compute the band-pass signals as per Clause 5.1.3 to 5.1.4 of
    ECMA-418-2, 2nd Ed. (2022).

    Calculation of the band-pass signals along the 53 critical band rates
    scale.

    Parameters
    ----------
    signal: numpy.array
        'Pa', time signal values. The sampling frequency of the signal must be 48000 Hz.
    
    Returns
    -------
    block_bandpass_signals: list of numpy.array
        Band-pass signals
    """

    # ************************************************************************
    # Sections 5.1.3 of ECMA-418-2, 2nd Ed. (2022)
    
    # OUTER AND MIDDLE EAR FILTERING (5.1.3)
    sos_ear = _ear_filter_design()
    signal_filtered = sp_signal.sosfilt(sos_ear, sig, axis=0)


    # ************************************************************************
    # Section 5.1.4 of ECMA-418-2, 2nd Ed. (2022)
    
    # AUDITORY FILTERING BANK (5.1.4)

    # Order of the Outer and Middle ear filter
    filter_order_k = 5
    
    # Sampling frequency
    fs = 48000.00
    
    # Auditory filters centre frequencies
    centre_freq = _auditory_filters_centre_freq()

    block_bandpass_signals = []
    for band_number in range(53):
        
        bm_mod, am_mod = _gammatone(centre_freq[band_number],
                                    k=filter_order_k,
                                    fs=fs)

        # Eq. (12)
        band_pass_signal = 2.0*sp_signal.lfilter(bm_mod,
                                                 am_mod, 
                                                 signal_filtered,
                                                 axis=0).real
        
        block_bandpass_signals.append(band_pass_signal)
        

    return block_bandpass_signals